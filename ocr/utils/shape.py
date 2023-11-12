import cv2
import numpy as np


def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-5
    return v[1] / l


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def find_bottom(pts):

    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.875:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            dist_list = []
            if len(candidate) > 2:

                bottom_idx = np.argsort([angle for s1, s2, angle in candidate])[0:2]
                bottoms = [candidate[bottom_idx[0]][:2], candidate[bottom_idx[1]][0:2]]
                long_edge1, long_edge2 = find_long_edges(pts, bottoms)
                edge_length1 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge1]
                edge_length2 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge2]
                l1 = sum(edge_length1)
                l2 = sum(edge_length2)
                len1 = len(edge_length1)
                len2 = len(edge_length2)

                if l1 > 2*l2 or l2 > 2*l1 or len1 == 0 or len2 == 0:
                    for i in range(len(pts)):
                        mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                        mid_list.append((i, (i + 1) % len(pts), mid_point))

                    for i in range(len(pts)):
                        for j in range(len(pts)):
                            s1, e1, mid1 = mid_list[i]
                            s2, e2, mid2 = mid_list[j]
                            dist = norm2(mid1 - mid2)
                            dist_list.append((s1, e1, s2, e2, dist))
                    bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-1]
                    bottoms = [dist_list[bottom_idx][:2], dist_list[bottom_idx][2:4]]
            else:
                mid_list = []
                for i in range(len(pts)):
                    mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                    mid_list.append((i, (i + 1) % len(pts), mid_point))

                dist_list = []
                for i in range(len(pts)):
                    for j in range(len(pts)):
                        s1, e1, mid1 = mid_list[i]
                        s2, e2, mid2 = mid_list[j]
                        dist = norm2(mid1 - mid2)
                        dist_list.append((s1, e1, s2, e2, dist))
                bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
                bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]
    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
        # bottoms = [(0, 1), (2, 3)] if 2 * d1 < d2 and d1 > 32 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while i % n_pts != b2_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while i % n_pts != b1_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_edge_seqence(points, n_parts):
    pts_num = points.shape[0]
    long_edge = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def get_sample_point(text_mask, num_points, approx_factor, scales=None):
    # get sample point in contours
    contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon = approx_factor * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
    # approx = contours[0].reshape((-1, 2))
    if scales is None:
        ctrl_points = split_edge_seqence(approx, num_points)
    else:
        ctrl_points = split_edge_seqence(approx*scales, num_points)
    ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

    return ctrl_points