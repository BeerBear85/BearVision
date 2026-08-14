package dk.bearvision.app

data class UserProfile(
    val name: String,
    val email: String,
    val serverUrl: String,
)

data class VideoItem(
    val jobId: String,
    val capturedAt: String?,
    val durationSeconds: Double?,
)

data class VideoPage(
    val displayName: String,
    val videos: List<VideoItem>,
)
