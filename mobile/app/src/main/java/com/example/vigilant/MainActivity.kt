package com.example.vigilant

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.OrientationEventListener
import android.view.Surface
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.android.synthetic.main.activity_main.*
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

const val MODEL_NAME = "mobile_segnet.pt"

class MainActivity : AppCompatActivity() {
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    var module: Module? = null

    //    Utilities for module loading

    fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        try {
            context.assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        } catch (e: IOException) {
            Log.e("pytorchandroid", "Error process asset $assetName to file path")
        }
        return null
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)



        cameraProviderFuture = ProcessCameraProvider.getInstance(this)


        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(this))

        //  Switches the Content View


    }

    fun bindPreview(cameraProvider : ProcessCameraProvider) {

        val imageCapture = ImageCapture.Builder().build()


        // Needed for model predictions
        val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(1280,720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

        // Might be needed to change orientation from portrait to landscape
        val orientationEventListener = object : OrientationEventListener(this as Context){
            override fun onOrientationChanged(orientation: Int) {
                val rotation : Int = when (orientation) {
                    in 45..134 -> Surface.ROTATION_270
                    in 135..224 -> Surface.ROTATION_180
                    in 225..314 -> Surface.ROTATION_90
                    else -> Surface.ROTATION_0
                }
                imageCapture.targetRotation = rotation
            }
  }
        orientationEventListener.enable()


        // Preview needed for visuals (camera)
        var preview : Preview = Preview.Builder()
                .setTargetRotation(Surface.ROTATION_270)
                .build()
        //  Activates Back Camera
        var cameraSelector : CameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

        preview.setSurfaceProvider(previewView.surfaceProvider)

        // Retrieve Model Paths and loads it
       val modulePATH: String = File(assetFilePath(this, MODEL_NAME)).absolutePath
        module = Module.load(modulePATH)


        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageCapture, imageAnalysis)
    }

}