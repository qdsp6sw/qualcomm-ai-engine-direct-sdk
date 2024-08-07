/*
 * Copyright (c) 2024 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */

package com.qualcomm.qti.psnpedemo.utils;
import android.util.Log;
import java.lang.Math;
import java.util.Arrays;

public class Mat {
    public enum InterType {
        NEAREST,
        LINEAR
    }
    private final int[] m_dims;
    private final int[] m_strides;
    private final float[] m_data;

    public Mat(float[] data, int... dims) {
        checkData(data, dims);
        m_dims = new int[dims.length];
        System.arraycopy(dims, 0, m_dims, 0, m_dims.length);
        m_strides = calStrides(m_dims);
        m_data = data;
    }

    public Mat(int... dims) {
        checkDims(dims);
        m_dims = new int[dims.length];
        System.arraycopy(dims, 0, m_dims, 0, m_dims.length);
        m_strides = calStrides(m_dims);
        m_data = new float[dims[0] * m_strides[0]];
    }

    public float get(int...idx) {
        checkIndexBound(idx);
        int p = calDataPos(idx);
        return m_data[p];
    }

    public void set(float value, int... idx) {
        checkIndexBound(idx);
        int p = calDataPos(idx);
        m_data[p] = value;
    }

    public void set(Mat m, int... idx) {
        assert m.rank() == this.rank();
        int rank = m.rank();
        int[] d_idx = Arrays.copyOf(idx, rank);
        int[] s_idx = new int[rank];
        int[] s_dims = m.m_dims;
        boolean[] carry = new boolean[rank - 1];
        int r = rank - 1;
        while (r >= 0) {
            if (r < rank - 1) {
                if (carry[r]) {
                    carry[r] = false;
                    ++r;
                }
                else if (s_idx[r] == s_dims[r] - 1) {
                    s_idx[r] = 0;
                    d_idx[r] = idx[r];
                    carry[r] = true;
                    --r;
                }
                else {
                    s_idx[r] += 1;
                    d_idx[r] += 1;
                    ++r;
                }
                continue;
            }
            this.set(m.get(s_idx), d_idx);
            s_idx[r] += 1;
            d_idx[r] += 1;
            if (s_idx[r] == s_dims[r]) {
                s_idx[r] = 0;
                d_idx[r] = idx[r];
                --r;
            }
        }
    }

    public int[] dims() {
        int[] dims = new int[m_dims.length];
        System.arraycopy(m_dims, 0, dims, 0, dims.length);
        return dims;
    }

    public int rank() {
        return m_dims.length;
    }

    public int size() {
        return m_data.length;
    }

    public float[] data(boolean copy) {
        if (copy) {
            float[] dst_data = new float[m_data.length];
            System.arraycopy(m_data, 0, dst_data, 0, dst_data.length);
            return dst_data;
        }
        else {
            return m_data;
        }
    }

    public Mat copy() {
        return new Mat(data(true), dims());
    }

    public Mat reshape(int... d_dims) {
        return new Mat(m_data, d_dims);
    }

    public Mat reFormat(int... axis_order) {
        final int rank = this.rank();
        assert axis_order.length == rank;
        int[] d_dims = new int[rank];
        int[] d_idx = new int[rank];
        int[] s_idx = new int[rank];
        for (int i = 0; i < rank; i++) {
            d_dims[i] = m_dims[axis_order[i]];
        }
        Mat d_mat = new Mat(d_dims);

        boolean[] carry = new boolean[rank - 1];
        int r = rank - 1;
        while (r >= 0) {
            if (r < rank - 1) {
                if (carry[r]) {
                    carry[r] = false;
                    ++r;
                }
                else if (d_idx[r] == d_dims[r] - 1) {
                    s_idx[axis_order[r]] = 0;
                    d_idx[r] = 0;
                    carry[r] = true;
                    --r;
                }
                else {
                    s_idx[axis_order[r]] += 1;
                    d_idx[r] += 1;
                    ++r;
                }
                continue;
            }
            d_mat.set(this.get(s_idx), d_idx);
            s_idx[axis_order[r]] += 1;
            d_idx[r] += 1;
            if (d_idx[r] == d_dims[r]) {
                s_idx[axis_order[r]] = 0;
                d_idx[r] = 0;
                --r;
            }
        }
        return d_mat;
    }

    public int calDataPos(int...idx) {
        int p = 0;
        for (int i = 0; i < idx.length; ++i) {
            p += idx[i] * m_strides[i];
        }
        return p;
    }

    public static Mat arange(int range) {
        float[] data = new float[range];
        for (int i = 0; i < range; ++i) {
            data[i] = i;
        }
        return new Mat(data, range);
    }

    public static Mat[] mashgrid(Mat x, Mat y) {
        assert x.rank() == 1;
        assert y.rank() == 1;
        Mat grid_x = new Mat(y.size(), x.size());
        Mat grid_y = new Mat(y.size(), x.size());
        for (int raw = 0; raw < grid_x.dims()[0]; ++raw) {
            for (int col = 0; col < grid_x.dims()[1]; ++col) {
                grid_x.set(x.get(col), raw, col);
                grid_y.set(y.get(raw), raw, col);
            }
        }
        return new Mat[] {grid_x, grid_y};
    }

    public Mat[] split(int axis, int... sections) {
        axis = axis < 0 ? rank() + axis : axis;
        int[] lengths = new int[sections.length];
        int[] starts = new int[sections.length];
        int[][] div_dims = new int[sections.length][];
        for (int s = 0; s < sections.length; ++s) {
            starts[s]  = s == 0 ? 0 : starts[s - 1] + sections[s - 1];
            lengths[s] = sections[s] * m_strides[axis];
            div_dims[s] = dims();
            div_dims[s][axis] = sections[s];
        }
        Mat[] div_mat_list = new Mat[sections.length];
        int step = 0 == axis ? size() : m_strides[axis - 1];
        for (int s = 0; s < sections.length; ++s) {
            Mat div_mat = new Mat(div_dims[s]);
            float[] dst_data = div_mat.data(false);
            int src_start = starts[s] * m_strides[axis];
            for (int i = src_start, j = 0; i < size(); i += step, j += lengths[s]) {
                System.arraycopy(m_data, i, dst_data, j, lengths[s]);
            }
            div_mat_list[s] = div_mat;
        }
        return div_mat_list;
    }

    public static Mat merge(Mat... mats){
        assert mats.length >= 2;
        for (int i = 1; i < mats.length; ++i) {
            assert Arrays.equals(mats[0].m_dims, mats[i].m_dims);
        }
        int d_rank = mats[0].rank() + 1;
        int[] d_dims = new int[d_rank];
        d_dims[0] = mats.length;
        System.arraycopy(mats[0].m_dims, 0, d_dims, 1, d_rank - 1);
        Mat d_mat = new Mat(d_dims);
        int size = mats[0].size();
        for (int i = 0; i < mats.length; ++i) {
            System.arraycopy(mats[i].m_data, 0, d_mat.m_data, i * size, size);
        }
        return d_mat;
    }

    public static Mat concat(int axis, Mat... s_mat) {
        axis = axis < 0 ? s_mat[0].rank() + axis : axis;
        int rank = s_mat[0].rank();
        assert axis < rank;
        int[] d_dims = s_mat[0].dims();
        for (int i = 1; i < s_mat.length; ++i) {
            assert s_mat[i].rank() == rank;
            for (int j = 0; j < rank; ++j) {
                assert j == axis || s_mat[0].m_dims[j] == s_mat[i].m_dims[j];
            }
            d_dims[axis] += s_mat[i].m_dims[axis];
        }

        int[] reformat_order = new int[rank];
        for (int i = 0; i < rank; ++i) {
            reformat_order[i] = i;
        }
        reformat_order[0] = axis;
        reformat_order[axis] = 0;

        int[] d_reformat_dims = Arrays.copyOf(d_dims, rank);
        d_reformat_dims[0] = d_dims[axis];
        d_reformat_dims[axis] = d_dims[0];

        Mat d_mat = new Mat(d_reformat_dims);
        for (int i = 0, pos = 0; i < s_mat.length; ++i) {
            Mat s_reformat_mat = s_mat[i].reFormat(reformat_order);
            System.arraycopy(s_reformat_mat.m_data, 0, d_mat.m_data, pos, s_reformat_mat.size());
            pos += s_reformat_mat.size();
        }
        d_mat = d_mat.reFormat(reformat_order);
        return d_mat;
    }

    public static Mat sigmoid(Mat in) {
        Mat out = new Mat(in.dims());
        for (int i = 0; i < in.size(); ++i) {
            double x = in.m_data[i];
            out.m_data[i]  = 1 / (1 + (float)Math.pow(Math.E, -1 * x));
        }
        return out;
    }

    public static Mat exp(Mat in) {
        Mat out = new Mat(in.dims());
        for (int i = 0; i < in.size(); ++i) {
            double x = in.m_data[i];
            out.m_data[i] = (float) Math.pow(Math.E, x);
        }
        return out;
    }

    public static Mat add(Mat m1, Mat m2) {
        Mat mat1 = m1.rank() <= m2.rank() ? m1 : m2;
        Mat mat2 = m1.rank() > m2.rank() ? m1 : m2;
        if (mat1.rank() < mat2.rank()) {
            int[] new_dims = new int[mat2.rank()];
            Arrays.fill(new_dims, 1);
            System.arraycopy(mat1.m_dims, 0, new_dims, mat2.rank() - mat1.rank(), mat1.rank());
            mat1 = mat1.reshape(new_dims);
        }
        boolean mat1_need_broadcast = false, mat2_need_broadcast = false;
        int[] broadcast_dims = mat1.dims();
        for (int i = broadcast_dims.length - 1; i >= 0; --i) {
            if (mat1.m_dims[i] > mat2.m_dims[i]) {
                if (mat2.m_dims[i] != 1) {
                    throw new IllegalArgumentException(String.format("%s can not broadcast with %s",
                            Arrays.toString(mat1.m_dims), Arrays.toString(mat2.m_dims)));
                }
                mat2_need_broadcast = true;
            }
            else if (mat1.m_dims[i] < mat2.m_dims[i]) {
                if (mat1.m_dims[i] != 1) {
                    throw new IllegalArgumentException(String.format("%s can not broadcast with %s",
                            Arrays.toString(mat1.m_dims), Arrays.toString(mat2.m_dims)));
                }
                mat1_need_broadcast = true;
                broadcast_dims[i] = mat2.m_dims[i];
            }
        }
        if (mat1_need_broadcast) {
            mat1 = mat1.broadcast(broadcast_dims);
        }
        if (mat2_need_broadcast) {
            mat2 = mat2.broadcast(broadcast_dims);
        }

        Mat result = new Mat(mat1.m_dims);
        for (int i = 0; i < result.size(); ++i) {
            result.m_data[i] = mat1.m_data[i] + mat2.m_data[i];
        }
        return result;
    }

    public static Mat mul(Mat m1, Mat m2) {
        Mat mat1 = m1.rank() <= m2.rank() ? m1 : m2;
        Mat mat2 = m1.rank() > m2.rank() ? m1 : m2;
        if (mat1.rank() < mat2.rank()) {
            int[] new_dims = new int[mat2.rank()];
            Arrays.fill(new_dims, 1);
            System.arraycopy(mat1.m_dims, 0, new_dims, mat2.rank() - mat1.rank(), mat1.rank());
            mat1 = mat1.reshape(new_dims);
        }
        boolean mat1_need_broadcast = false, mat2_need_broadcast = false;
        int[] broadcast_dims = mat1.dims();
        for (int i = broadcast_dims.length - 1; i >= 0; --i) {
            if (mat1.m_dims[i] > mat2.m_dims[i]) {
                if (mat2.m_dims[i] != 1) {
                    throw new IllegalArgumentException(String.format("%s can not broadcast with %s",
                            Arrays.toString(mat1.m_dims), Arrays.toString(mat2.m_dims)));
                }
                mat2_need_broadcast = true;
            }
            else if (mat1.m_dims[i] < mat2.m_dims[i]) {
                if (mat1.m_dims[i] != 1) {
                    throw new IllegalArgumentException(String.format("%s can not broadcast with %s",
                            Arrays.toString(mat1.m_dims), Arrays.toString(mat2.m_dims)));
                }
                mat1_need_broadcast = true;
                broadcast_dims[i] = mat2.m_dims[i];
            }
        }
        if (mat1_need_broadcast) {
            mat1 = mat1.broadcast(broadcast_dims);
        }
        if (mat2_need_broadcast) {
            mat2 = mat2.broadcast(broadcast_dims);
        }

        Mat result = new Mat(mat1.m_dims);
        for (int i = 0; i < result.size(); ++i) {
            result.m_data[i] = mat1.m_data[i] * mat2.m_data[i];
        }
        return result;
    }

    private Mat broadcast(int... d_dims) {
        final int rank = rank();
        assert d_dims.length == rank;
        int[] broadcast = new int[rank];
        for (int i = 0; i < rank; ++i) {
            broadcast[i] = d_dims[i] / m_dims[i];
        }

        Mat d_mat = copy();
        int[] order = new int[rank];
        for (int i = 0; i < order.length; ++i) {
            order[i] = i;
        }
        for (int i = 0; i < broadcast.length; ++i) {
            if (broadcast[i] == 1) {
                continue;
            }
            order[i] = 0;
            order[0] = i;
            d_mat = d_mat.reFormat(order);
            int[] tmp_dims = Arrays.copyOf(d_mat.m_dims, rank);
            tmp_dims[0] = broadcast[i];
            Mat tmp = new Mat(tmp_dims);
            for (int j = 0; j < broadcast[i]; ++j) {
                System.arraycopy(d_mat.m_data, 0, tmp.m_data, j * d_mat.size(), d_mat.size());
            }
            d_mat = tmp.reFormat(order);
            order[0] = 0;
            order[i] = i;
        }
        return d_mat;
    }

    public static Mat div(Mat m, float value) {
        Mat result = new Mat(m.dims());
        for (int i = 0; i < m.size(); ++i) {
            result.m_data[i] = m.m_data[i] / value;
        }
        return result;
    }

    public static Mat resize(Mat s_mat, InterType interpolation, int...d_dims) {
        assert s_mat.rank() == 2;
        assert d_dims.length == 2;
        if (interpolation == InterType.NEAREST) {
            return resize_nearest(s_mat, d_dims);
        }
        else if (interpolation == InterType.LINEAR){
            return resize_liner(s_mat, d_dims);
        }
        else {
            throw new IllegalArgumentException("Unsupported InterType: " + interpolation);
        }
    }

    private static Mat resize_nearest(Mat s_mat, int...d_dims) {
        Mat d_mat = new Mat(d_dims);
        int s_height = s_mat.m_dims[0], s_width = s_mat.m_dims[1];
        int d_height = d_dims[0], d_width = d_dims[1];
        double scale_h = 1.0 / ((double)d_height / s_height);
        double scale_w = 1.0 / ((double)d_width / s_width);
        for (int d_h = 0; d_h < d_height; ++d_h) {
            for (int d_w = 0; d_w < d_width; ++d_w) {
                int s_h = (int)(d_h * scale_h);
                int s_w = (int)(d_w * scale_w);
                d_mat.set(s_mat.get(s_h, s_w), d_h, d_w);
            }
        }
        return d_mat;
    }

    private static Mat resize_liner(Mat s_mat, int...d_dims) {
        Mat d_mat = new Mat(d_dims);
        int s_height = s_mat.m_dims[0], s_width = s_mat.m_dims[1];
        int d_height = d_dims[0], d_width = d_dims[1];

        double scale_y = (double)s_height / (double)d_height;
        double scale_x = (double)s_width / (double)d_width;
        for(int dy = 0; dy < d_height; ++dy){
            double fy = ((double)dy + 0.5) * scale_y - 0.5;
            int sy = (int)fy;
            fy -= sy;
            if(sy < 0){
                fy = 0.0; sy = 0;
            }
            if(sy >= s_height - 1){
                fy = 0.0; sy = s_height - 2;
            }
            for(int dx = 0; dx < d_width; ++dx){
                double fx = ((double)dx + 0.5) * scale_x - 0.5;
                int sx = (int)fx;
                fx -= sx;
                if(sx < 0){
                    fx = 0.0; sx = 0;
                }
                if(sx >= s_width - 1){
                    fx = 0.0; sx = s_width - 2;
                }
                double d_value = (1.0-fx) * (1.0-fy) * s_mat.get(sy, sx)
                               + fx * (1.0-fy) * s_mat.get(sy, sx+1)
                               + (1.0-fx) * fy * s_mat.get(sy+1, sx)
                               + fx * fy * s_mat.get(sy+1, sx+1);
                d_mat.set((float)d_value, dy, dx);
            }
        }
        return d_mat;
    }

    private static int[] calStrides(int[] dims) {
        int[] strides = new int[dims.length];
        int stride = 1;
        strides[dims.length - 1] = stride;
        for (int i = dims.length - 1 ; i > 0; --i) {
            stride *= dims[i];
            strides[i - 1] = stride;
        }
        return strides;
    }

    private void checkData(float[] data, int[] dims) {
        checkDims(dims);
        assert data != null;
        int buf_size = dims[0];
        for (int i = 1; i < dims.length; ++i) {
            buf_size *= dims[i];
        }
        assert data.length == buf_size;
    }

    private void checkDims(int[] dims) {
        assert dims != null;
        for (int d : dims) {
            assert d > 0;
        }
    }

    private void checkIndexBound(int... idx) {
        if (idx.length > rank()) {
            throw new IndexOutOfBoundsException(
                    String.format("Index %s is out of rank %d ", Arrays.toString(idx), rank()));
        }
        for (int i = 0; i < idx.length; ++i) {
            if (idx[i] < 0 || idx[i] >= m_dims[i]) {
                throw new IndexOutOfBoundsException(
                        String.format("Index %s is out of dims %s ",
                                Arrays.toString(idx), Arrays.toString(m_dims)));
            }
        }
    }

    public void tofile(String path) {
        Util.writeArrayTofile(path, m_data, true);
    }

    public void debugMatInfo(String mat_name) {
        String dim_str = arrayToString(m_dims);
        String stride_str = arrayToString(m_strides);
        Log.d("debug", String.format("Mat[%s]:(dims:%s, strides:%s, size:%d)", mat_name, dim_str, stride_str, size()));
    }
    static String arrayToString(int[] a) {
        StringBuilder str = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            str.append(a[i]).append(i == a.length - 1 ? "]" : ",");
        }
        return str.toString();
    }

    public float[][] to2dArray() {
        float[][] data = new float[m_dims[0]][m_dims[1]];
        for (int i = 0; i < m_dims[0]; i++) {
            for (int j = 0; j < m_dims[1]; j++) {
                data[i][j] = get(i, j);
            }
        }
        return data;
    }
}
