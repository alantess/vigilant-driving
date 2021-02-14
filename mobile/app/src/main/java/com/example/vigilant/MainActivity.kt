package com.example.vigilant

import android.annotation.SuppressLint
import android.app.Application
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.OrientationEventListener
import android.view.Surface
import android.widget.Toast
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
import kotlinx.android.synthetic.main.visuals.*
import kotlinx.android.synthetic.main.visuals.view.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService

const val MODEL_NAME = "mobile_segnet.pt"

class MainActivity : AppCompatActivity() {

    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    var module: Module? = null
    private lateinit var cameraExecutor: ExecutorService
    private var mInputTensor: Tensor? = null
    private var outputImg: Tensor? = null
    private var mInputTensorBuffer: FloatBuffer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        visuals.setOnClickListener {
            setContentView(R.layout.visuals)
            startActivity()
        }
    }
    // Starts the camera
    fun startActivity(){
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()
            bindPreview(cameraProvider)
        }, ContextCompat.getMainExecutor(this))

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

        val imageCapture = ImageCapture.Builder().build()

        // Preview needed for visuals (camera)
        var preview : Preview = Preview.Builder()
            .setTargetRotation(Surface.ROTATION_0)
            .build()
        //  Activates Back Camera
        var cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(previewView.surfaceProvider)

        // Might be needed to change orientation from portrait to landscape
        val orientationEventListener = object : OrientationEventListener(this as Context){
            override fun onOrientationChanged(orientation: Int) {
                val rotation : Int = when (orientation) {
                    in 45..134 -> Surface.ROTATION_0
                    in 135..224 -> Surface.ROTATION_0
                    in 225..314 -> Surface.ROTATION_0
                    else -> Surface.ROTATION_0
                }
                imageCapture.targetRotation = rotation

            }
        }
        orientationEventListener.enable()


        // Needed for model predictions
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(256,256))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(previewView.display.rotation)
            .build()


        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), ImageAnalysis.Analyzer { image ->
            if(module == null){
                // Retrieve Model Paths and loads it
                val modulePATH: String = File(assetFilePath(this, MODEL_NAME)).absolutePath
                module = Module.load(modulePATH)
                Toast.makeText(applicationContext, "Module Loaded", Toast.LENGTH_LONG).show()
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

            // Commit a forward pass through the network (1x3x256x256)
            outputImg = module?.forward(IValue.from(mInputTensor))?.toTensor()
            Log.d("OUTPUT TENSOR", outputImg.toString())

        })


        cameraProvider.unbindAll()

        cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageCapture, imageAnalysis)
    }

}