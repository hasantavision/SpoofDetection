package com.amartha.spooflibrary;

import android.graphics.Bitmap;
import android.graphics.Rect;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.util.ArrayList;
import java.util.List;

public class FaceDetectorProcessor {
    private final FaceDetector faceDetector;
    public interface FaceDetectionListener {
        void onFaceDetectionComplete(List<Rect> boundingBoxes);
    }
    public FaceDetectorProcessor() {
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                        .build();
        faceDetector = FaceDetection.getClient(options);
    }

    public void detectFaces(Bitmap bitmap, final FaceDetectionListener listener) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        Task<List<Face>> task = faceDetector.process(image);
        task.addOnCompleteListener(new OnCompleteListener<List<Face>>() {
            @Override
            public void onComplete(Task<List<Face>> task) {
                List<Rect> boundingBoxes = new ArrayList<>();
                if (task.isSuccessful()) {
                    List<Face> faces = task.getResult();
                    for (Face face : faces) {
                        boundingBoxes.add(face.getBoundingBox());
                    }
                }
                listener.onFaceDetectionComplete(boundingBoxes);
            }
        });
    }
}

