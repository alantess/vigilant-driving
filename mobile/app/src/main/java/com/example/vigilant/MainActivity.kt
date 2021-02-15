package com.example.vigilant

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.visuals.*
import kotlinx.android.synthetic.main.visuals.view.*
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

const val MODEL_NAME = "mobile_segnet.pt"

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    private var imageCapture: ImageCapture? = null
    private lateinit var outputDirectory: File
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    var module: Module? = null
    private lateinit var cameraExecutor: ExecutorService
    private var mInputTensor: Tensor? = null
    private var outputImg: Tensor? = null
    private var mInputTensorBuffer: FloatBuffer? = null
    private var take_photo_button: Button? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        outputDirectory = getOutputDirectory()

        cameraExecutor = Executors.newSingleThreadExecutor()



        visuals.setOnClickListener {
            setContentView(R.layout.visuals)
            startActivity()
            camera_capture_button.setOnClickListener { takePhoto() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
    // Starts the camera
    fun startActivity(){
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(this))

    }
    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() } }
        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    fun takePhoto() {
        val imageCapture = imageCapture ?: return
        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(FILENAME_FORMAT, Locale.US
            ).format(System.currentTimeMillis()) + ".jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions, ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e("PHOTO", "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = Uri.fromFile(photoFile)
                    val msg = "Photo capture succeeded: $savedUri"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d("PHOTO", msg)
                }
            })



    }
    // Retrieve Model path
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

    // Retrieves the images and makes a prediction
    @SuppressLint("UnsafeExperimentalUsageError")
    fun bindPreview(cameraProvider : ProcessCameraProvider) {
        // Preview needed for visuals (camera)
        var preview : Preview = Preview.Builder()
            .setTargetResolution(Size(1280,720))
            .setTargetRotation(Surface.ROTATION_0)
            .build()
        //  Activates Back Camera
        var cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(previewView.surfaceProvider)

        // Used for taking a photo
        imageCapture = ImageCapture.Builder()
            .build()



        // Needed for model predictions
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(256,256))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(previewView.display.rotation)
            .build()


        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), ImageAnalysis.Analyzer { image ->

            if(module == null){
                progressBar.visibility = View.VISIBLE
                // Retrieve Model Paths and loads it
                val modulePATH: String = File(assetFilePath(this, MODEL_NAME)).absolutePath
                module = Module.load(modulePATH)
                progressBar.visibility = View.GONE
            }
            val rotationDegrees = image.imageInfo.rotationDegrees
            // Allocate Memory for image
            mInputTensorBuffer =
                Tensor.allocateFloatBuffer(3 *256 * 256)
            mInputTensor = Tensor.fromBlob(
                mInputTensorBuffer,
                longArrayOf(1, 3, 256,256)
            )
            // Turn image into Tensor
            TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                image.image, rotationDegrees,
                256, 256,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                mInputTensorBuffer, 0)

            // Commits a forward pass through the network (1x3x256x256) and turns image into a float
//            outputImg = module?.forward(IValue.from(mInputTensor))?.toTensor()
//            val outputs: FloatArray? = outputImg?.dataAsFloatArray
//            Log.d("OUTPUT TENSOR", outputImg.toString())


            // Float to Image View


            image.close()

        })


        cameraProvider.unbindAll()

        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageCapture, imageAnalysis)
    }

}
