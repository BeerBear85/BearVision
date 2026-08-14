package dk.bearvision.app

import android.util.Patterns
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.ui.PlayerView
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val colors = darkColorScheme(
    primary = Color(0xFFFFB45A),
    onPrimary = Color(0xFF352000),
    background = Color(0xFF17130F),
    surface = Color(0xFF211B16),
    onBackground = Color(0xFFFFF8F1),
    onSurface = Color(0xFFFFF8F1),
)

@Composable
fun BearVisionApp(viewModel: BearVisionViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            viewModel.dismissError()
        }
    }

    MaterialTheme(colorScheme = colors) {
        Box(Modifier.fillMaxSize()) {
            when {
                state.initializing -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                state.selectedVideo != null && state.profile != null -> PlayerScreen(
                    profile = state.profile!!,
                    video = state.selectedVideo!!,
                    onBack = viewModel::closeVideo,
                )
                state.profile == null -> ProfileScreen(onContinue = viewModel::signIn)
                else -> VideoListScreen(
                    profile = state.profile!!,
                    videos = state.videos,
                    loading = state.loading,
                    onRefresh = viewModel::refresh,
                    onOpen = viewModel::openVideo,
                    onLogOut = viewModel::logOut,
                )
            }
            SnackbarHost(snackbar, Modifier.align(Alignment.BottomCenter))
        }
    }
}

@Composable
private fun ProfileScreen(onContinue: (String, String, String) -> Unit) {
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var server by remember { mutableStateOf("") }
    val valid = name.isNotBlank() && Patterns.EMAIL_ADDRESS.matcher(email.trim()).matches() &&
        server.isNotBlank()

    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 56.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("BearVision", style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))
        Text("Se dine egne optagelser fra det lokale BearVision-system.")
        Spacer(Modifier.height(32.dp))
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            label = { Text("Navn") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlinedTextField(
            value = email,
            onValueChange = { email = it },
            label = { Text("E-mail") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlinedTextField(
            value = server,
            onValueChange = { server = it },
            label = { Text("Serveradresse") },
            placeholder = { Text("http://192.168.1.50:4321") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(24.dp))
        Button(
            onClick = { onContinue(name, email, server) },
            enabled = valid,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Vis mine videoer") }
        Spacer(Modifier.height(12.dp))
        Text(
            "Prototype: Der bruges endnu ikke password eller e-mailbekræftelse.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.65f),
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VideoListScreen(
    profile: UserProfile,
    videos: List<VideoItem>,
    loading: Boolean,
    onRefresh: () -> Unit,
    onOpen: (VideoItem) -> Unit,
    onLogOut: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Hej ${profile.name}") },
                actions = { TextButton(onClick = onLogOut) { Text("Skift bruger") } },
            )
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text("Dine videoer", style = MaterialTheme.typography.headlineSmall)
                OutlinedButton(onClick = onRefresh, enabled = !loading) { Text("Opdater") }
            }
            if (loading && videos.isEmpty()) {
                Box(Modifier.fillMaxSize()) { CircularProgressIndicator(Modifier.align(Alignment.Center)) }
            } else if (videos.isEmpty()) {
                Box(Modifier.fillMaxSize()) {
                    Text(
                        "Der er endnu ingen videoer til ${profile.email}.",
                        modifier = Modifier.align(Alignment.Center),
                    )
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(videos, key = { it.jobId }) { video ->
                        VideoCard(video = video, onClick = { onOpen(video) })
                    }
                    item { Spacer(Modifier.height(24.dp)) }
                }
            }
        }
    }
}

@Composable
private fun VideoCard(video: VideoItem, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(Modifier.padding(18.dp)) {
            Text(formatTimestamp(video.capturedAt), fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(6.dp))
            Text(
                video.durationSeconds?.let { "Varighed: ${it.toInt()} sekunder" }
                    ?: "Varighed ukendt",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            )
            Spacer(Modifier.height(12.dp))
            Text("Afspil video", color = MaterialTheme.colorScheme.primary)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PlayerScreen(profile: UserProfile, video: VideoItem, onBack: () -> Unit) {
    val context = LocalContext.current
    val player = remember(profile, video.jobId) {
        val api = BearVisionApi(profile)
        val dataSource = DefaultHttpDataSource.Factory()
            .setDefaultRequestProperties(api.requestHeaders())
        val mediaSource = ProgressiveMediaSource.Factory(dataSource)
            .createMediaSource(MediaItem.fromUri(api.videoUrl(video.jobId)))
        ExoPlayer.Builder(context).build().apply {
            setMediaSource(mediaSource)
            prepare()
            playWhenReady = true
        }
    }
    DisposableEffect(player) { onDispose { player.release() } }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(formatTimestamp(video.capturedAt)) },
                navigationIcon = { TextButton(onClick = onBack) { Text("Tilbage") } },
            )
        },
    ) { padding ->
        AndroidView(
            factory = { PlayerView(it).apply { this.player = player } },
            update = { it.player = player },
            modifier = Modifier.fillMaxSize().padding(padding),
        )
    }
}

private fun formatTimestamp(value: String?): String {
    if (value == null) return "Video"
    return runCatching {
        DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm")
            .withZone(ZoneId.systemDefault())
            .format(Instant.parse(value))
    }.getOrDefault(value)
}
