package com.amartha.spooflibrary;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class FaceFeaturesModel {
    private ObjectDetector objectDetector;
    public interface FaceFeaturesListener {
        void onFaceFeaturesComplete(List<RectF> boundingBoxes, List<String> labels);
        void onFaceFeaturesFailed(Exception e);
    }

    public FaceFeaturesModel(Context context) throws IOException {
        AssetManager assetManager = context.getAssets();

        File file = new File(context.getCacheDir() + "/model.tflite");

        try (InputStream inputStream = assetManager.open("modelfusion.tflite");
             FileOutputStream outputStream = new FileOutputStream(file)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        ObjectDetectorOptions options = ObjectDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().build())
                .setMaxResults(4)
                .setScoreThreshold(0.4f)
                .build();

        objectDetector = ObjectDetector.createFromFileAndOptions(
                file, options);
    }

    public void detectFeatures(Bitmap bitmap, final FaceFeaturesListener listener) {
        AsyncTask.execute(new Runnable() {
            public void run() {
                TensorImage image = TensorImage.fromBitmap(bitmap);
                List<Detection> detections = objectDetector.detect(image);
                List<RectF> boundingBoxes = new ArrayList<>();
                List<String> labels = new ArrayList<>();

                for(Detection detection : detections) {
                    Log.d("======================================DETECTION", detection.getCategories().get(0).getLabel());
                    RectF boundingBox = detection.getBoundingBox();
                    String label = detection.getCategories().get(0).getLabel();
                    boundingBoxes.add(boundingBox);
                    labels.add(label);
                }

                if (listener != null) {
                    listener.onFaceFeaturesComplete(boundingBoxes, labels);
                }
            }});
    }
}
