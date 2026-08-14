package dk.bearvision.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI

class BearVisionApi(private val profile: UserProfile) {
    private val baseUrl = normalizeServerUrl(profile.serverUrl)

    suspend fun listVideos(): VideoPage = withContext(Dispatchers.IO) {
        val payload = getJson("/api/app/videos?page=1&pageSize=100")
        val root = JSONObject(payload)
        val user = root.getJSONObject("user")
        val items = root.getJSONArray("items")
        val videos = buildList {
            for (index in 0 until items.length()) {
                val item = items.getJSONObject(index)
                add(
                    VideoItem(
                        jobId = item.getString("jobId"),
                        capturedAt = item.optString("captureStartedAt").ifBlank { null },
                        durationSeconds = if (item.has("durationSeconds")) {
                            item.getDouble("durationSeconds")
                        } else null,
                    ),
                )
            }
        }
        VideoPage(user.getString("displayName"), videos)
    }

    fun videoUrl(jobId: String): String {
        require(jobId.matches(Regex("[A-Za-z0-9._:-]+"))) { "Ugyldigt video-id" }
        return "$baseUrl/api/app/videos/$jobId/video"
    }

    fun requestHeaders(): Map<String, String> = mapOf(EMAIL_HEADER to profile.email)

    private fun getJson(path: String): String {
        val connection = URI.create(baseUrl + path).toURL().openConnection() as HttpURLConnection
        return try {
            connection.requestMethod = "GET"
            connection.connectTimeout = 8_000
            connection.readTimeout = 20_000
            connection.setRequestProperty(EMAIL_HEADER, profile.email)
            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (status !in 200..299) {
                val message = runCatching { JSONObject(body).optString("error") }.getOrNull()
                val localized = when (message) {
                    "user not found" -> "E-mailen findes ikke i BearVision"
                    "video not found for user" -> "Videoen tilhører ikke denne bruger"
                    else -> message?.ifBlank { null }
                }
                throw ApiException(localized ?: "Serveren svarede med $status")
            }
            body
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        const val EMAIL_HEADER = "X-BearVision-Email"

        fun normalizeServerUrl(value: String): String {
            val normalized = value.trim().trimEnd('/')
            val uri = runCatching { URI.create(normalized) }.getOrNull()
                ?: throw IllegalArgumentException("Serveradressen er ugyldig")
            if (uri.scheme !in setOf("http", "https") || uri.host.isNullOrBlank()) {
                throw IllegalArgumentException("Brug en fuld adresse, fx http://192.168.1.50:4321")
            }
            return normalized
        }
    }
}

class ApiException(message: String) : Exception(message)
