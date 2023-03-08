package com.amartha.spooflibrary;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier.ImageClassifierOptions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;


public class SpoofDetectorProcessor {
    private ImageClassifier imageClassifier;

    public interface SpoofDetectionListener {
        void onSpoofDetectionComplete(boolean isSpoof);

        void onSpoofDetectionFailed(Exception e);
    }

    public SpoofDetectorProcessor(Context context) throws IOException {
        AssetManager assetManager = context.getAssets();

        File modelFile = new File(context.getCacheDir() + "/model_screen.tflite");

        try (InputStream inputStream = assetManager.open("fusion.tflite");
             FileOutputStream outputStream = new FileOutputStream(modelFile)) {

            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        ImageClassifierOptions options = ImageClassifierOptions.builder()
                .setBaseOptions(BaseOptions.builder().build())
                .setMaxResults(1)
                .build();

        imageClassifier = ImageClassifier.createFromFileAndOptions(
                modelFile, options);
    }

    public void detectSpoof(Bitmap bitmap, SpoofDetectionListener listener) {
        AsyncTask.execute(new Runnable() {
            public void run() {
                TensorImage tensorImage = new TensorImage();
                tensorImage.load(bitmap);
                List<Classifications> classifications = imageClassifier.classify(tensorImage);
                for (Classifications classification : classifications) {
                    Log.d("=============SpoofDetectorProcessor", classification.toString());
                }
                boolean isSpoof = classifications.get(0).getCategories().get(0).getLabel().equals("spoof");
                if(listener != null) {
                    listener.onSpoofDetectionComplete(isSpoof);
                }
            }
        });
    }

}

