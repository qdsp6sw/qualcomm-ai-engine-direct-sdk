/*
 * Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.os.Trace;
import android.util.Log;
import android.util.Pair;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.DetectionResult;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.utils.BoundingBox;
import com.qualcomm.qti.psnpedemo.utils.ComputeUtil;
import com.qualcomm.qti.psnpedemo.utils.Mat;
import com.qualcomm.qti.psnpedemo.utils.MathUtils;
import com.qualcomm.qti.psnpedemo.utils.Util;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class DetectionPostprocessor extends PostProcessor {
    //private final String debug_path = "/sdcard/Android/data/com.qualcomm.qti.psnpedemo/files/output_psnpedemo";
    private final String TAG = "DetectionPostprocessor";
    private final String model_name;
    private final String data_set;
    private String gt_path;
    private String yolov3_coco_map_path;
    private double map;
    private final float iou_threshold = (float)0.5;

    public DetectionPostprocessor(EvaluationCallBacks evaluationCallBacks, ModelInfo modelInfo, int imageNumber) {
        super(imageNumber);
        this.evaluationCallBacks = evaluationCallBacks;
        model_name = modelInfo.getModelName();
        data_set = modelInfo.getDataSetName();
        if(Objects.equals(data_set, "voc")){
            String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
            gt_path = packagePath + "/datasets/" + data_set + "/voc_annotations.json";
        }
        else if(Objects.equals(data_set, "coco")){
            String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
            gt_path = packagePath + "/datasets/" + data_set + "/coco_annotations.json";
            yolov3_coco_map_path = packagePath + "/datasets/" + data_set + "/coco_map.json";
        }

    }

    @Override
    public boolean postProcessResult(ArrayList<File> inputImages) {
        Trace.beginSection("postProcessResult");
        Log.d(TAG, "start into detection post process!");
        Map<String, Map<Integer, List<BoundingBox>>> gt = new HashMap<>();
        Map<String, Map<Integer, List<BoundingBox>>> pred = new HashMap<>();
        Map<Integer, Integer> gt_count = new HashMap<>();
        Map<Integer, Integer> pred_count = new HashMap<>();
        List<String> img_ids = new ArrayList<>();
        Map<String, int[]> img_shapes = new HashMap<>();
        for(int j = 0;j < inputImages.size(); j++){
            String img_name = inputImages.get(j).getName();
            String image_id = img_name.split("\\.")[0].replaceAll("^(0+)", "");
            img_ids.add(j, image_id);
        }
        loadGT(gt_path, img_ids, gt, gt_count, img_shapes);
        int[] coco_map = new int[0];
        if (model_name.contains("yolov3")) {
            coco_map = loadCocoMap(yolov3_coco_map_path);
        }

        for(int i = 0; i < inputImages.size(); ++i) {
            String img_id = img_ids.get(i);
            Map<String, float[]> output = readOutput(i);
            int[] img_shape = img_shapes.get(img_id);
            if (model_name.contains("yolov3")) {
                getYoloV3Result(output, img_id, img_shape, coco_map, pred, pred_count);
            }
            else if (model_name.contains("ssd")) {
                getSsdResult(output, img_id, img_shape, pred, pred_count);
            }
            else {
                Log.e(TAG,"model " + model_name + " is not supported");
                return false;
            }
        }

        map = 0;
        for(Integer class_id : pred_count.keySet()) {
            if(!gt_count.containsKey(class_id)){
                continue;
            }
            double ap = getAveragePrecision(class_id, gt, pred, gt_count, pred_count, iou_threshold);
            Log.d(TAG, String.format("class %d map: %f", class_id, ap));
            map += ap;
        }
        map = map / pred_count.size();

        Trace.endSection();
        return true;
    }

    @Override
    public void setResult(Result result) {
        DetectionResult dresult= (DetectionResult)result;
        dresult.setMap(map);
    }

    @Override
    public void resetResult(){}

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }

    private void loadGT(String gt_path, List<String> img_ids,
                        Map<String, Map<Integer, List<BoundingBox>>> gt,
                        Map<Integer, Integer> gt_count,
                        Map<String, int[]> img_shapes) {
        try {
            String json_str = Util.readStringFromFile(gt_path);
            JSONObject json_object = new JSONObject(json_str);
            JSONArray json_annotations = json_object.getJSONArray("annotations");
            for (int i = 0; i < json_annotations.length(); ++i) {
                JSONObject json_item = json_annotations.getJSONObject(i);
                String img_id = json_item.getString("image_id");
                if (!img_ids.contains(img_id)) {
                    continue;
                }
                Integer class_id = json_item.getInt("category_id");
                JSONArray json_bbox = json_item.getJSONArray("bbox");
                BoundingBox bbox = new BoundingBox((float)json_bbox.getDouble(0),
                        (float)json_bbox.getDouble(1),
                        (float)(json_bbox.getDouble(0) + json_bbox.getDouble(2)),
                        (float)(json_bbox.getDouble(1) + json_bbox.getDouble(3)));
                if (!gt.containsKey(img_id)) {
                    gt.put(img_id, new HashMap<>());
                }
                if (!gt.get(img_id).containsKey(class_id)) {
                    gt.get(img_id).put(class_id, new ArrayList<>());
                }
                gt.get(img_id).get(class_id).add(bbox);
                gt_count.put(class_id, gt_count.getOrDefault(class_id, 0) + 1);
            }

            for (String img_id : img_ids) {
                img_shapes.put(img_id, new int[]{0, 0});
            }
            JSONArray json_images = json_object.getJSONArray("images");
            for (int i = 0; i < json_images.length(); ++i) {
                JSONObject json_item = json_images.getJSONObject(i);
                String img_id = json_item.getString("id");
                if (!img_ids.contains(img_id)) {
                    continue;
                }
                int height = json_item.getInt("height");
                int width = json_item.getInt("width");
                img_shapes.get(img_id)[0] = height;
                img_shapes.get(img_id)[1] = width;
            }
        }
        catch (Exception e) {
            Log.e(TAG,String.format("Fail to load ground truth. file:%s. error:%s", gt_path, e));
            throw new RuntimeException(e);
        }
    }

    private int[] loadCocoMap(String coco_map_path) {
        try {
            String json_str = Util.readStringFromFile(yolov3_coco_map_path);
            JSONObject json_coco_map = new JSONObject(json_str);
            int class_num = json_coco_map.length();
            int[] coco_map = new int[class_num];
            for (Iterator<String> it = json_coco_map.keys(); it.hasNext(); ) {
                String yolov3_class_str = it.next();
                int yolov3_class = Integer.parseInt(yolov3_class_str);
                int coco_class = json_coco_map.getInt(yolov3_class_str);
                coco_map[yolov3_class] = coco_class;
            }
            return coco_map;
        }
        catch (Exception e) {
            Log.e(TAG,String.format("Fail to load coco map. file:%s. error:%s", coco_map_path, e));
            throw new RuntimeException(e);
        }
    }

    public void getYoloV3Result(Map<String, float[]> output,
                                String img_id,
                                int[] img_shape,
                                int[] coco_map,
                                Map<String, Map<Integer, List<BoundingBox>>> pred,
                                Map<Integer, Integer> pred_count) {
        int class_num = coco_map.length;
        float[]  raw_data_13 = output.get("yolov3/yolov3_head/Conv_6/Conv2D:0");
        float[]  raw_data_26 = output.get("yolov3/yolov3_head/Conv_14/Conv2D:0");
        float[]  raw_data_52 = output.get("yolov3/yolov3_head/Conv_22/Conv2D:0");
        int[] feature_13_dims = {13, 13, 255};
        Mat feature_13 = new Mat(raw_data_13, feature_13_dims);
        int[] feature_26_dims = {26, 26, 255};
        Mat feature_26 = new Mat(raw_data_26, feature_26_dims);
        int[] feature_52_dims = {52, 52, 255};
        Mat feature_52 = new Mat(raw_data_52, feature_52_dims);
        ArrayList<BoundingBox> predicts = yolov3Prediction(feature_13, feature_26, feature_52, class_num);

        Map<Integer, List<BoundingBox>> predictionData = new HashMap<>();
        for (int j = 0; j < predicts.size(); ++j) {
            Integer class_id = coco_map[predicts.get(j).class_id] + 1;
            //img_shape: [height, width];
            float y1 = predicts.get(j).y1 * img_shape[0];
            float x1 = predicts.get(j).x1 * img_shape[1];
            float y2 = predicts.get(j).y2 * img_shape[0];
            float x2 = predicts.get(j).x2 * img_shape[1];
            BoundingBox bbox = new BoundingBox(x1, y1, x2, y2);
            if(!predictionData.containsKey(class_id)) {
                predictionData.put(class_id, new ArrayList<>());
            }
            predictionData.get(class_id).add(bbox);

            if (!pred.containsKey(img_id)) {
                pred.put(img_id, new HashMap<>());
            }
            if (!pred.get(img_id).containsKey(class_id)) {
                pred.get(img_id).put(class_id, new ArrayList<>());
            }
            pred.get(img_id).get(class_id).add(bbox);

            pred_count.put(class_id, pred_count.getOrDefault(class_id, 0) + 1);
        }
    }

    public void
    getSsdResult(Map<String, float[]> output,
                 String img_id,
                 int[] img_shape,
                 Map<String, Map<Integer, List<BoundingBox>>> pred,
                 Map<Integer, Integer> pred_count) {
        Map<Integer, List<BoundingBox>> predictionData = new HashMap<>();
        float[]  bboxArray = null;
        float[]  scoreArray = null;
        float[]  classArrayTmp = null;
        for(String key: output.keySet()) {
            if(key.contains("boxes") && bboxArray == null){
                bboxArray = output.get(key);
            }
            else if(key.contains("scores") && scoreArray == null){
                scoreArray = output.get(key);
            }
            else if(key.contains("classes") && classArrayTmp == null){
                classArrayTmp = output.get(key);
            }
        }
        if(bboxArray == null || scoreArray == null || classArrayTmp == null){
            Log.e(TAG,"can't find all outputs layer");
            return;
        }

        classArrayTmp = MathUtils.round(classArrayTmp);
        int[] classArray = new int[classArrayTmp.length];
        if(data_set.equals("coco")){
            for(int k = 0; k < classArray.length; k++) {
                classArray[k] = (int)classArrayTmp[k] + 1;
            }
        }
        for(int j = 0; j < scoreArray.length; j++) {
            if(scoreArray[j] == 0) {
                break;
            }
            if(scoreArray[j] < 0) {
                continue;
            }
            //img_shape: [height, width];
            float y1 = (bboxArray[(j * 4)]) * img_shape[0];
            float x1 = (bboxArray[(j * 4) + 1]) * img_shape[1];
            float y2 = (bboxArray[(j * 4) + 2]) * img_shape[0];
            float x2 = (bboxArray[(j * 4) + 3]) * img_shape[1];
            Integer class_id = classArray[j];

            BoundingBox bbox = new BoundingBox(class_id, scoreArray[j], x1, y1, x2, y2);
            if(!predictionData.containsKey(class_id)) {
                predictionData.put(class_id, new ArrayList<>());
            }
            predictionData.get(class_id).add(bbox);

            if (!pred.containsKey(img_id)) {
                pred.put(img_id, new HashMap<>());
            }
            if (!pred.get(img_id).containsKey(class_id)) {
                pred.get(img_id).put(class_id, new ArrayList<>());
            }
            pred.get(img_id).get(class_id).add(bbox);

            pred_count.put(class_id, pred_count.getOrDefault(class_id, 0) + 1);
        }
    }

    public double
    getAveragePrecision(Integer class_id,
                        Map<String, Map<Integer, List<BoundingBox>>> gt,
                        Map<String, Map<Integer, List<BoundingBox>>> pred,
                        Map<Integer, Integer> gt_count,
                        Map<Integer, Integer> pred_count,
                        float iou_threshold) {
        List<Pair<Boolean, Float>> matched = new ArrayList<>();
        for(String img_id : pred.keySet()) {
            List<BoundingBox> gt_data = gt.getOrDefault(img_id, new HashMap<>()).getOrDefault(class_id, new ArrayList<>());
            List<BoundingBox> pred_data = pred.get(img_id).getOrDefault(class_id, new ArrayList<>());
            boolean[] used_gt_box = new boolean[gt_data.size()];
            for (int i = 0; i < pred_data.size(); ++i) {
                float match_iou = 0;
                int match_gt_idx = 0;
                for (int j = 0; j < gt_data.size(); ++j) {
                    if (used_gt_box[j]) {
                        continue;
                    }
                    float iou = pred_data.get(i).iou(gt_data.get(j));
                    if(Float.compare(iou, match_iou) == 1) {
                        match_iou = iou;
                        match_gt_idx = j;
                    }
                }
                if (match_iou > iou_threshold) {
                    matched.add(Pair.create(Boolean.TRUE, pred_data.get(i).score));
                    used_gt_box[match_gt_idx] = true;
                }
                else {
                    matched.add(Pair.create(Boolean.FALSE, pred_data.get(i).score));
                }
            }
        }

        matched.sort((t0, t1) -> Float.compare(t1.second, t0.second));
        int[] TP = new int[matched.size()];
        int[] FP = new int[matched.size()];
        TP[0] = matched.get(0).first ? 1 : 0;
        FP[0] = matched.get(0).first ? 0 : 1;
        for (int i = 1; i < matched.size(); ++i) {
            TP[i] = TP[i - 1] + (matched.get(i).first ? 1 : 0);
            FP[i] = FP[i - 1] + (matched.get(i).first ? 0 : 1);
        }

        float eps = (float)2.220446049250313e-16;
        float[] rec = new float[TP.length];
        float[] prec = new float[TP.length];
        for(int i = 0; i < TP.length; ++i) {
            rec[i] = (float)TP[i] / gt_count.get(class_id);
            prec[i] = (float)TP[i] / (FP[i] + TP[i] + eps);
        }

        double ap = 0;
        if(Objects.equals(data_set, "voc")){
            ap = getVocAp(rec, prec);
        }
        else if (Objects.equals(data_set, "coco")){
            ap = getCocoAp(rec, prec);
        }
        return ap;
    }

    public static float getVocAp(float[] rec, float[] prec) {
        float ap = (float)0.0;
        float[] mrec = new float[rec.length + 2];
        float[] mprec = new float[prec.length + 2];
        mrec[0] = 0;
        System.arraycopy(rec, 0, mrec, 1, rec.length);
        mrec[mrec.length -1] = 1;
        mprec[0] = 0;
        System.arraycopy(prec, 0, mprec, 1, prec.length);
        mprec[mprec.length -1] = 0;
        for(int i = mprec.length - 2; i >=0; i--) {
            mprec[i] = Math.max(mprec[i], mprec[i+1]);
        }
        List<Integer> iList = new ArrayList<>();
        for(int i = 1; i < mrec.length; i++) {
            if(mrec[i] != mrec[i-1]) {
                iList.add(i);
            }
        }
        for(int i: iList) {
            ap += (mrec[i]-mrec[i-1]) * mprec[i];
        }
        return ap;
    }

    public static float getCocoAp(float[] rec, float[] prec) {
        float ap;
        double[] recThrs = new double[101];
        for(int i = 0; i < recThrs.length; i++) {
            recThrs[i] = (1.0 * i)/100;
        }
        int nd = prec.length;
        double[] q = new double[recThrs.length];
        double[] precTmp = new double[prec.length + 1];
        for(int i = 0; i < prec.length; i++) {
            precTmp[i] = prec[i];
        }
        for(int i = nd - 1; i >0; i--) {
            if(precTmp[i] > precTmp[i - 1]) {
                precTmp[i - 1] = precTmp[i];
            }
        }
        precTmp[precTmp.length - 1] = prec[prec.length - 1];
        int[] inds = new int[recThrs.length];
        for(int i = 0; i < inds.length; i++) {
            inds[i] = ComputeUtil.searchSorted(rec, recThrs[i]);
        }
        int count = 0;
        for(int i:inds) {
            try {
                q[count++] = precTmp[i];
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        ap = (float) ComputeUtil.getAverage(q);
        return ap;
    }

    private ArrayList<BoundingBox>
    yolov3Prediction(Mat feature_13, Mat feature_26, Mat feature_52, int class_num) {
        int[] input_size = {PSNPEManager.getInputDimensions()[1],
                            PSNPEManager.getInputDimensions()[2]};
        int[][] anchors13 = {{116, 90}, {156, 198}, {373, 326}};
        int[][] anchors26 = {{30, 61}, {62, 45}, {59, 119}};
        int[][] anchors52 = {{10, 13}, {16, 30}, {33, 23}};
        // feature_13[13, 13, 255]
        HashMap<String, Mat> result_13 = parse_data(feature_13, anchors13, input_size, class_num);
        // feature_26[26, 26, 255]
        HashMap<String, Mat> result_26 = parse_data(feature_26, anchors26, input_size, class_num);
        // feature_52[52, 52, 255]
        HashMap<String, Mat> result_52 = parse_data(feature_52, anchors52, input_size, class_num);
        Mat boxes = result_13.get("boxes");
        boxes = Mat.concat(0, boxes, result_26.get("boxes"));
        boxes = Mat.concat(0, boxes, result_52.get("boxes"));
        Mat confs = result_13.get("confs");
        confs = Mat.concat(0, confs, result_26.get("confs"));
        confs = Mat.concat(0, confs, result_52.get("confs"));
        Mat probs = result_13.get("probs");
        probs = Mat.concat(0, probs, result_26.get("probs"));
        probs = Mat.concat(0, probs, result_52.get("probs"));

        int box_num = boxes.dims()[0];
        for (int i = 0; i < box_num; ++i) {
            float center_x = boxes.get(i, 0);
            float center_y = boxes.get(i, 1);
            float width = boxes.get(i, 2);
            float height = boxes.get(i, 3);
            boxes.set(center_x - width / 2, i, 0); // x_min
            boxes.set(center_y - height / 2, i, 1); // y_min
            boxes.set(center_x + width / 2, i, 2); // x_max
            boxes.set(center_y + height / 2, i, 3); // y_max
        }

        Mat scores = Mat.mul(confs, probs);

        float score_thresh = (float)0.3;
        ArrayList<BoundingBox> filtered_boxes = new ArrayList<>();
        for (int i = 0; i < class_num; i++) {
            ArrayList<BoundingBox> class_boxes = new ArrayList<>();
            for (int j = 0; j < box_num; j++) {
                float score = scores.get(j, i);
                if (score >= score_thresh) {
                    class_boxes.add(new BoundingBox(i, score, boxes.get(j, 0), boxes.get(j, 1),
                                                    boxes.get(j, 2), boxes.get(j, 3)));
                }
            }
            ArrayList<BoundingBox> picked_boxes = MathUtils.nms(class_boxes, 0.45);
            filtered_boxes.addAll(picked_boxes);
        }

        ArrayList<BoundingBox> final_boxes = new ArrayList<>();
        for (int i = 0; i < filtered_boxes.size(); ++i) {
            BoundingBox box = filtered_boxes.get(i);
            float y1 = box.y1 / input_size[0];
            float y2 = box.y2 / input_size[0];
            float x1 = box.x1 / input_size[1];
            float x2 = box.x2 / input_size[1];
            final_boxes.add(new BoundingBox(box.class_id, box.score, x1, y1, x2, y2));
        }

        return final_boxes;
    }

    private HashMap<String, Mat>
    parse_data(Mat featureMap, int[][] anchors, int[] img_size, int classNum) {
        int[] grid_size = {featureMap.dims()[0], featureMap.dims()[1]}; //size in [h, w] format
        Mat ratio = new Mat(new float[]{(float)img_size[1]/(float)grid_size[1],
                                        (float)img_size[0]/(float)grid_size[0]},
                            2);
        Mat rescaled_anchors = new Mat(anchors.length, 2);
        for (int i = 0; i < anchors.length; ++i) {
            rescaled_anchors.set(anchors[i][0] / ratio.get(0), i, 0);
            rescaled_anchors.set(anchors[i][1] / ratio.get(1), i , 1);
        }
        featureMap = featureMap.reshape(grid_size[0], grid_size[1], 3, 5 + classNum);
        Mat[] div_feature = featureMap.split(-1, 2, 2, 1, classNum);
        Mat box_centers = div_feature[0];
        Mat box_sizes = div_feature[1];
        Mat conf_logits = div_feature[2];
        Mat prob_logits = div_feature[3];

        Mat[] grids_mashed = Mat.mashgrid(Mat.arange(grid_size[0]), Mat.arange(grid_size[1]));
        Mat x_offset = grids_mashed[0].reshape(grids_mashed[0].size(), 1);
        Mat y_offset = grids_mashed[1].reshape(grids_mashed[1].size(), 1);
        Mat x_y_offset = Mat.concat(-1, x_offset, y_offset);
        x_y_offset = x_y_offset.reshape(grid_size[0], grid_size[1], 1, 2);

        box_centers = Mat.sigmoid(box_centers);
        box_centers = Mat.add(box_centers, x_y_offset);
        box_centers = Mat.mul(box_centers, ratio);
        box_sizes = Mat.exp(box_sizes);
        box_sizes = Mat.mul(box_sizes, rescaled_anchors);
        box_sizes = Mat.mul(box_sizes, ratio);
        Mat boxes = Mat.concat(-1, box_centers, box_sizes);

        boxes = boxes.reshape(grid_size[0] * grid_size[1] * 3, 4);
        Mat confs = Mat.sigmoid(conf_logits);
        confs = confs.reshape(grid_size[0] * grid_size[1] * 3, 1);
        Mat probs = Mat.sigmoid(prob_logits);
        probs = probs.reshape(grid_size[0] * grid_size[1] * 3, classNum);
        HashMap<String, Mat> ret = new HashMap<>();
        ret.put("boxes", boxes);
        ret.put("confs", confs);
        ret.put("probs", probs);
        return ret;
    }
}