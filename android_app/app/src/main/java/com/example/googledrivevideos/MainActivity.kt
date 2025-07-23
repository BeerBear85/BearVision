package com.example.googledrivevideos

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.api.client.extensions.android.http.AndroidHttp
import com.google.api.client.googleapis.auth.oauth2.GoogleCredential
import com.google.api.client.json.gson.GsonFactory
import com.google.api.services.drive.Drive
import com.google.api.services.drive.model.FileList

class MainActivity : AppCompatActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: VideoAdapter
    private lateinit var driveService: Drive

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recyclerView = findViewById(R.id.video_list)
        recyclerView.layoutManager = LinearLayoutManager(this)
        adapter = VideoAdapter { file ->
            Thread { downloadFile(file.id, file.name) }.start()
        }
        recyclerView.adapter = adapter

        Thread {
            setupDriveService()
            loadVideos()
        }.start()
    }

    private fun setupDriveService() {
        val json = resources.openRawResource(R.raw.service_account)
        val credential = GoogleCredential.fromStream(json)
            .createScoped(listOf("https://www.googleapis.com/auth/drive.readonly"))
        driveService = Drive.Builder(
            AndroidHttp.newCompatibleTransport(),
            GsonFactory.getDefaultInstance(),
            credential
        ).setApplicationName("DriveVideos").build()
    }

    private fun loadVideos() {
        try {
            val request = driveService.files().list()
                .setQ("mimeType contains 'video/' and trashed = false")
                .setFields("files(id, name)")
            val result: FileList = request.execute()
            val files = result.files ?: emptyList()
            runOnUiThread { adapter.submitList(files) }
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread { Toast.makeText(this, "Failed to list files", Toast.LENGTH_LONG).show() }
        }
    }

    private fun downloadFile(fileId: String, fileName: String) {
        try {
            val output = openFileOutput(fileName, MODE_PRIVATE)
            driveService.files().get(fileId).executeMediaAndDownloadTo(output)
            output.close()
            runOnUiThread { Toast.makeText(this, "Downloaded $fileName", Toast.LENGTH_SHORT).show() }
        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread { Toast.makeText(this, "Download failed", Toast.LENGTH_LONG).show() }
        }
    }
}
