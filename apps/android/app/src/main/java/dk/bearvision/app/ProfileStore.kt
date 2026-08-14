package dk.bearvision.app

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.profileDataStore by preferencesDataStore(name = "bearvision_profile")

class ProfileStore(private val context: Context) {
    private object Keys {
        val name = stringPreferencesKey("name")
        val email = stringPreferencesKey("email")
        val serverUrl = stringPreferencesKey("server_url")
    }

    val profile: Flow<UserProfile?> = context.profileDataStore.data.map { values ->
        val name = values[Keys.name]
        val email = values[Keys.email]
        val serverUrl = values[Keys.serverUrl]
        if (name == null || email == null || serverUrl == null) null
        else UserProfile(name, email, serverUrl)
    }

    suspend fun save(profile: UserProfile) {
        context.profileDataStore.edit { values ->
            values[Keys.name] = profile.name
            values[Keys.email] = profile.email
            values[Keys.serverUrl] = profile.serverUrl
        }
    }

    suspend fun clear() {
        context.profileDataStore.edit { it.clear() }
    }
}
