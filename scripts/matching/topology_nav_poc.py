from __future__ import annotations

import time
from typing import Dict

from tqdm import tqdm

from matching.generate_test_data import get_aruco, create_matrix_front_patch, apply_rotations, extract_patch
from matching.topology_nav_graph import default_path
from scripts.matching.topology_nav_plot import *
from scripts.matching.topology_nav_utils import *
from scripts.utils.datasets import get_dataset_by_name, load_paths_and_files, resize_image
from scripts.utils.norlab_sync_viz import seq1

logger = logging.getLogger(__name__)


def create_norlab_graph() -> Graph:
    g = Graph()
    n1 = Node("1", np.array([3, 7]), 5)
    n2 = Node("2", np.array([10, 18]), 3)
    g.add_edge(n1, n2)
    n3 = Node("3", np.array([-1, 25.5]), 3)
    g.add_edge(n2, n3)
    n4 = Node("4", np.array([-25, 35]), 4)
    g.add_edge(n3, n4)
    n5 = Node("5", np.array([-32, 26]), 3)
    g.add_edge(n4, n5)
    n6 = Node("6", np.array([-40, 10]), 3)
    g.add_edge(n5, n6)
    n7 = Node("7", np.array([-8, -4]), 3)
    g.add_edge(n6, n7)
    g.add_edge(n7, n1)
    n8 = Node("8", np.array([2, -12]), 4)
    g.add_edge(n7, n8)
    n9 = Node("9", np.array([-10, 14]), 3)
    g.add_edge(n1, n9)
    g.add_edge(n9, n5)
    n10 = Node("10", np.array([7, 32]), 3)
    g.add_edge(n2, n10)
    g.add_edge(n3, n10)
    n11 = Node("11", np.array([19, 11]), 3)
    g.add_edge(n2, n11)
    n12 = Node("12", np.array([21, 35]), 3)
    g.add_edge(n11, n12)
    n13 = Node("13", np.array([-16, 47]), 3)
    g.add_edge(n12, n13)
    g.add_edge(n13, n4)
    return g


def generate_scouting_data(g: Graph, gnss: pd.DataFrame, vis: bool):
    is_running = False
    for filepath, image in tqdm(load_paths_and_files(get_dataset_by_name(seq1 / "aerial" / "images"))):
        aruco = get_aruco(image)
        if aruco is None:
            """
            The aruco is required when using the norlab datasets as there is no estimate of the heading 
            of the uav. This is compensated by using the orientation of the aruco tag.
            In the final algorithm, this trick won't be necessary.
            """
            continue
        aruco = aruco[0, :, :]  # there is only one aruco tag
        aruco_a = np.arctan2(aruco[0, 1] - aruco[1, 1],
                             aruco[1, 0] - aruco[0, 0]) - np.pi / 2  # angle from ugv to uav
        aruco_c = np.mean(aruco, axis=0)  # position of the aruco in the image
        pixels_per_meter = np.sum(np.array([np.linalg.norm(aruco[i, :] - aruco[i + 1, :])
                                            for i in range(-1, 3)])) / 1.4  # resolution of the image

        uav_time = int(filepath.stem)
        search_gnss = gnss.iloc[(gnss['timestamp'] - uav_time).abs().argmin()]
        tf_ugv_to_map = get_transformation_matrix(np.array([search_gnss['x'], search_gnss['y']]),
                                                  search_gnss['yaw'])
        tf_map_to_ugv = np.linalg.inv(tf_ugv_to_map)

        patches_width = 480  # 480 ~= pixels_per_meter*2
        patches = [p + [0, -90] for p in create_matrix_front_patch(patches_width,
                                                                   image.shape[:2],
                                                                   pattern=(4, 7),
                                                                   margin=(20, 10))]
        patches = apply_rotations(patches, [np.rad2deg(-search_gnss['yaw'] + aruco_a)])
        w, h = 250, 325
        exclusion_rect = apply_rotations([np.array([aruco_c + [-w, -h], aruco_c + [w, -h],
                                                    aruco_c + [w, h], aruco_c + [-w, h]]) + [0, -100]],
                                         [3])[0].astype(np.int32)

        def filter_patch():
            for p in patches:
                coordinate_in_ugv = from_pixel_to_ugv(np.mean(p, axis=0), aruco_a, aruco_c, pixels_per_meter)
                coordinate = (tf_ugv_to_map @ np.array([coordinate_in_ugv[0], coordinate_in_ugv[1], 1]))[:2]
                n = g.get_current_node(coordinate)
                yield p, n, coordinate, n is not None and not overlap(p, exclusion_rect)

        for p, n, coordinate, valid in filter_patch():
            if valid:
                n.add_patch(extract_patch(p, image, patches_width), coordinate)

        if vis:
            background = image.copy()
            # draw aruco
            for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]):
                cv2.circle(image, aruco[i, :].astype(np.int32), radius=5, color=c, thickness=-1)

            # draw ugv/global frame
            for o, xy_list in zip([np.array([0, 0]),
                                   tf_map_to_ugv[:2, 2],
                                   np.array([0, 0])],
                                  [[(tf_map_to_ugv[:2, :2] @ np.array([1, 0])),
                                    (tf_map_to_ugv[:2, :2] @ np.array([0, 1]))],
                                   [(tf_map_to_ugv @ np.array([1, 0, 1]))[:2],
                                    (tf_map_to_ugv @ np.array([0, 1, 1]))[:2]],
                                   [np.array([1, 0]), np.array([0, 1])]
                                   ]):
                ij_o = from_ugv_to_pixel(o, aruco_a, aruco_c, pixels_per_meter)
                for xy, c in zip(xy_list, [(0, 0, 255), (0, 255, 0)]):
                    ij = from_ugv_to_pixel(xy, aruco_a, aruco_c, pixels_per_meter)
                    cv2.line(image, ij_o, ij, c, 10)

            # draw exclusion zone and patches
            cv2.polylines(image, [exclusion_rect], isClosed=True, color=(0, 0, 255), thickness=10)
            for p, n, coordinate, valid in filter_patch():
                c = (0, 255, 0) if valid else (0, 0, 255)
                for i, img in enumerate([image, background]):
                    if i == 0 or n is not None:
                        cv2.polylines(img, [p], isClosed=True, color=c, thickness=10)
                        cv2.putText(img,
                                    f"{n}",
                                    np.mean(p, axis=0).astype(np.int32) + [-200, -200],
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
                        cv2.putText(img,
                                    f"{coordinate[0]:.1f}",
                                    np.mean(p, axis=0).astype(np.int32) + [-200, -100],
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
                        cv2.putText(img,
                                    f"{coordinate[1]:.1f}",
                                    np.mean(p, axis=0).astype(np.int32) + [-200, 0],
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)

            # show image
            alpha = 0.5
            image_viz = cv2.addWeighted(resize_image(image, 600), alpha,
                                        resize_image(background, 600), 1 - alpha, 0)
            cv2.imshow("image", image_viz)
            k = cv2.waitKey(5 if is_running else 0) & 0xFF
            if k == ord('q'):
                break
            elif k == ord(' '):
                is_running = not is_running

    if not vis:
        for node in g.nodes:
            print(f"{node}: {len(node.patches)}")
            node.save_patches_metadata()
    else:
        cv2.destroyAllWindows()


def filter_scouting_data(g: Graph, keep_best: int, too_close_thresh: float):
    xfeat = get_xfeat()

    def median_score(_: str, __: np.ndarray, feats: Dict[str, np.ndarray]) -> float:
        return np.median(feats['scores'].cpu().numpy())

    for n in g.nodes:
        for c, _, path in tqdm(n.patches):
            image = cv2.imread(path)
            output = xfeat.detectAndCompute(image, top_k=4096)[0]
            output.update({'image_size': (image.shape[1], image.shape[0])})
            n.correspondance_data.append((path, c, output))

            if len(n.correspondance_data) > 1.5 * keep_best:
                n.correspondance_data = sorted(n.correspondance_data, key=lambda x: median_score(*x), reverse=True)
                correspondance_data_valid = [True for _ in range(len(n.correspondance_data))]
                for i, (path_i, c_i, feat_i) in enumerate(n.correspondance_data):
                    if not correspondance_data_valid[i]:
                        continue  # if already mark as too close, ignore it
                    for j, (path_j, c_j, feat_j) in enumerate(n.correspondance_data):
                        if not i < j:
                            continue
                        if np.linalg.norm(c_i - c_j) < too_close_thresh:
                            correspondance_data_valid[j] = False  # n.correspondance_data[j] too close to i
                n.correspondance_data = [d for d, v in zip(n.correspondance_data,
                                                           correspondance_data_valid) if v]
        n.correspondance_data = n.correspondance_data[:keep_best]
        n.save_correspondances()


def detect_ugv_location(graph: Graph, next_nodes: list[Node], current_nodes: list[Node],
                        gnss: pd.DataFrame, keep_best: int):
    is_running = False
    number_ugv_patch = 2
    patches_width = 650

    results = pd.DataFrame(columns=["current node", "next node", "ugv x", "ugv y", "naive is in node",
                                    "inference duration"]
                                   + number_ugv_patch * [str(i) for i in range(keep_best)])
    anim = RobotAnimator(graph.plot)

    for index, (next_node, current_node, (_, ugv_image), (filepath, ugv_bev)) in tqdm(
            enumerate(zip(next_nodes,
                          current_nodes,
                          load_paths_and_files(get_dataset_by_name(seq1 / "ground" / "images")),
                          load_paths_and_files(get_dataset_by_name(seq1 / "ground" / "projections"))))):
        if next_node is None:
            continue

        dt = time.time_ns()
        ugv_time = int(filepath.stem)
        search_gnss = gnss.iloc[(gnss['timestamp'] - ugv_time).abs().argmin()]
        ugv_2d_position = np.array([search_gnss['x'], search_gnss['y']])

        patches = create_matrix_front_patch(patches_width,
                                            ugv_bev.shape[:2],
                                            pattern=(1, number_ugv_patch),
                                            margin=(0, 400))
        yaw_ugv = search_gnss['yaw']  # angle from global to ugv
        patches = apply_rotations(patches, [np.rad2deg(-yaw_ugv)])
        extracted_patches = [extract_patch(p, ugv_bev, patches_width) for p in patches]

        is_in_node = False
        correspondances_each_pairs = []
        for extracted_patch in extracted_patches:
            xfeat = get_xfeat()
            ugv_feature = xfeat.detectAndCompute(extracted_patch, top_k=4096)[0]
            ugv_feature.update({'image_size': (extracted_patch.shape[1], extracted_patch.shape[0])})
            correspondances = []
            for img_path, c, uav_feature in next_node.correspondance_data:
                mkpts_0, mkpts_1, _ = xfeat.match_lighterglue(ugv_feature, uav_feature)
                correspondances.append((mkpts_0, mkpts_1))
                if not is_in_node and mkpts_0.shape[0] > 600:
                    is_in_node = True
            correspondances_each_pairs.append(correspondances)
        dt = time.time_ns() - dt
        anim.update_robot_pose(*ugv_2d_position, yaw_ugv)
        anim.update_node_display(next_node, np.array(correspondances_each_pairs))

        results.loc[len(results)] = [current_node, next_node, *ugv_2d_position, is_in_node, dt / 10 ** 9,
                                     *[mkpts_0.shape[0] for correspondances in correspondances_each_pairs
                                       for mkpts_0, _ in correspondances]]
        if index % 100 == 0:
            results.to_csv(str(default_path / "results.csv"), index=False)

        for p in patches:
            cv2.polylines(ugv_bev, [p], isClosed=True, color=(255, 255, 255), thickness=10)
            cv2.putText(ugv_bev,
                        f"{'-> ' + str(next_node) if current_node is None else 'at' + str(current_node)}",
                        np.mean(p, axis=0).astype(np.int32),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
        cv2.imshow("image", resize_image(ugv_bev, 600, 600))

        match_grid = generate_match_grid(next_node, ugv_2d_position, extracted_patches, correspondances_each_pairs)
        cv2.imshow("match_grid", resize_image(match_grid, 1500, 1500))

        k = cv2.waitKey(5 if is_running else 0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            is_running = not is_running

    results.to_csv(str(default_path / "results.csv"), index=False)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # TODO argparse
    keep_best = 10
    too_close_thresh = 1.8
    viz = False
    should_generate_scouting_data = True
    should_filter_scouting_data = True

    if (default_path / "graph.csv").exists():
        graph = Graph()
        graph.load()
    else:
        graph = create_norlab_graph()
        graph.save()

    # load position of robots
    gnss_norlab = pd.read_csv(
        get_dataset_by_name("norlab_ulaval_datasets/test_dataset/sequence1/rtk_odom/rtk_odom.csv"))
    # fix yaw angle
    real_angle = np.arctan2(gnss_norlab['y'][45] - gnss_norlab['y'][5],
                            gnss_norlab['x'][45] - gnss_norlab['x'][5])
    measured_angle = np.mean(gnss_norlab['yaw'][5:46])
    gnss_norlab['old_yaw'] = gnss_norlab['yaw']
    gnss_norlab['yaw'] = (gnss_norlab['yaw'] - measured_angle + real_angle) % np.pi

    # plot path
    # graph.plot()
    # for i, (x, y, yaw) in enumerate(zip(gnss_norlab['x'], gnss_norlab['y'], gnss_norlab['yaw'])):
    #     if i % 10 == 0:
    #         plt.arrow(x, y, 2 * np.cos(yaw), 2 * np.sin(yaw), head_width=0.5, color='g')
    # plot(gnss_norlab['x'], gnss_norlab['y'])
    # plt.show()

    if should_generate_scouting_data:
        generate_scouting_data(graph, gnss_norlab, viz)
    else:
        for n in graph.nodes:
            n.load_patches_metadata()

    if should_filter_scouting_data:
        filter_scouting_data(graph, keep_best, too_close_thresh)
    else:
        for n in graph.nodes:
            n.load_correspondances()

    next_nodes, current_nodes = get_path_in_node(gnss_norlab, graph)
    detect_ugv_location(graph, next_nodes, current_nodes, gnss_norlab, keep_best)


if __name__ == "__main__":
    main()
