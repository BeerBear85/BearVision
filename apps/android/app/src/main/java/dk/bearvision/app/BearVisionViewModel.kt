package dk.bearvision.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class BearVisionUiState(
    val initializing: Boolean = true,
    val loading: Boolean = false,
    val profile: UserProfile? = null,
    val videos: List<VideoItem> = emptyList(),
    val selectedVideo: VideoItem? = null,
    val error: String? = null,
)

class BearVisionViewModel(application: Application) : AndroidViewModel(application) {
    private val profileStore = ProfileStore(application)
    private val mutableState = MutableStateFlow(BearVisionUiState())
    val state: StateFlow<BearVisionUiState> = mutableState.asStateFlow()

    init {
        viewModelScope.launch {
            val stored = profileStore.profile.first()
            mutableState.update { it.copy(initializing = false, profile = stored) }
            if (stored != null) refresh()
        }
    }

    fun signIn(name: String, email: String, serverUrl: String) {
        val profile = runCatching {
            UserProfile(
                name = name.trim(),
                email = email.trim().lowercase(),
                serverUrl = BearVisionApi.normalizeServerUrl(serverUrl),
            )
        }.getOrElse { error ->
            mutableState.update { it.copy(error = error.message) }
            return
        }
        viewModelScope.launch {
            mutableState.update { it.copy(loading = true, error = null) }
            runCatching { BearVisionApi(profile).listVideos() }
                .onSuccess { page ->
                    profileStore.save(profile)
                    mutableState.update {
                        it.copy(
                            loading = false,
                            profile = profile,
                            videos = page.videos,
                            error = null,
                        )
                    }
                }
                .onFailure { error ->
                    mutableState.update {
                        it.copy(
                            loading = false,
                            profile = null,
                            error = error.message ?: "Kunne ikke kontakte serveren",
                        )
                    }
                }
        }
    }

    fun refresh() {
        val profile = mutableState.value.profile ?: return
        viewModelScope.launch {
            mutableState.update { it.copy(loading = true, error = null) }
            runCatching { BearVisionApi(profile).listVideos() }
                .onSuccess { page ->
                    mutableState.update {
                        it.copy(loading = false, videos = page.videos, error = null)
                    }
                }
                .onFailure { error ->
                    mutableState.update {
                        it.copy(
                            loading = false,
                            error = error.message ?: "Kunne ikke kontakte serveren",
                        )
                    }
                }
        }
    }

    fun openVideo(video: VideoItem) {
        mutableState.update { it.copy(selectedVideo = video) }
    }

    fun closeVideo() {
        mutableState.update { it.copy(selectedVideo = null) }
    }

    fun dismissError() {
        mutableState.update { it.copy(error = null) }
    }

    fun logOut() {
        viewModelScope.launch {
            profileStore.clear()
            mutableState.value = BearVisionUiState(initializing = false)
        }
    }
}
