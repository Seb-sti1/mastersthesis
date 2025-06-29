from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from scripts.matching.generate_test_data import create_matrix_front_patch, apply_rotations, extract_patch, scale
from scripts.matching.topology_nav_graph import default_path
from scripts.matching.topology_nav_plot import *
from scripts.matching.topology_nav_utils import *
from scripts.utils.datasets import get_dataset_by_name, load_paths_and_files, resize_image, load_paths

logger = logging.getLogger(__name__)

hub_drone = Path(get_dataset_by_name("rosbag_u2is/hub_drone_130625/"))


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


def create_u2is_graph() -> Graph:
    g = Graph()
    n1 = Node("1", np.array([2.33725, 48.59045]), 0.0001)
    n2 = Node("2", np.array([2.3378, 48.59025]), 0.0001)
    g.add_edge(n1, n2)
    n3 = Node("3", np.array([2.3381, 48.59015]), 0.0001)
    g.add_edge(n2, n3)
    n4 = Node("4", np.array([2.3384, 48.59005]), 0.0001)
    g.add_edge(n3, n4)
    n5 = Node("5", np.array([2.33894, 48.58985]), 0.0001)
    g.add_edge(n4, n5)
    n6 = Node("6", np.array([2.3391, 48.5902]), 0.0001)
    g.add_edge(n5, n6)
    n7 = Node("7", np.array([2.33825, 48.59046]), 0.0001)
    g.add_edge(n6, n7)
    return g


def generate_scouting_data(g: Graph, gnss: pd.DataFrame, vis: bool):
    for n in g.nodes:
        n.patches = []

    scaling = 2.8
    patches_width = 215

    is_running = False
    for filepath, image in tqdm(load_paths_and_files(hub_drone / "uav_rgb_gimbal")):
        uav_time = int(filepath.stem.split("_")[-1])
        search_gnss = gnss.iloc[(gnss['timestamp'] - uav_time).abs().argmin()]
        uav_coordinate = np.array([search_gnss['longitude'], search_gnss['latitude']])
        # TODO possible improvement export left/right patches
        patches = [(p - [0, image.shape[0] // 2 - patches_width // 2]).astype(np.int32) for p in
                   create_matrix_front_patch(patches_width,
                                             image.shape[:2],
                                             pattern=(1, 1),
                                             margin=(0, 0))]
        patches = apply_rotations(patches, [np.rad2deg(-search_gnss['yaw'])])
        for p in patches:
            n = g.get_current_node(uav_coordinate)
            if n is not None:
                n.add_patch(scale(extract_patch(p, image, patches_width), scaling), uav_coordinate,
                            str(filepath), np.mean(p, axis=0), patches_width, -1)
        if vis:
            background = image.copy()
            # draw exclusion zone and patches
            for p in patches:
                n = g.get_current_node(uav_coordinate)
                c = (0, 255, 0) if n is not None else (0, 0, 255)
                for i, img in enumerate([image, background]):
                    if i == 0 or n is not None:
                        cv2.polylines(img, [p], isClosed=True, color=c, thickness=10)
                        cv2.putText(img,
                                    f"{n}",
                                    np.mean(p, axis=0).astype(np.int32) + [-200, -200],
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
                        cv2.putText(img,
                                    f"{uav_coordinate[0]:.1f}",
                                    np.mean(p, axis=0).astype(np.int32) + [-200, -100],
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
                        cv2.putText(img,
                                    f"{uav_coordinate[1]:.1f}",
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

    if vis:
        cv2.destroyAllWindows()


def filter_scouting_data(g: Graph, keep_best: int, too_close_thresh: float):
    for n in g.nodes:
        n.correspondance_data = []

    xfeat = get_xfeat()

    def median_score(_: str, __: np.ndarray, feats: Dict[str, np.ndarray]) -> float:
        return np.median(feats['scores'].cpu().numpy())

    for n in g.nodes:
        for c, path, _, _, _, _ in tqdm(n.patches):
            image = cv2.imread(path)
            output = xfeat.detectAndCompute(image, top_k=4096)[0]
            output.update({'image_size': (image.shape[1], image.shape[0])})
            n.correspondance_data.append((path, c, output))

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


def detect_ugv_location(graph: Graph, next_nodes: list[Node], current_nodes: list[Node],
                        gnss: pd.DataFrame,
                        keep_best: int, match_count_thresh: int, match_count_probable_thresh: int,
                        viz: bool, save_to_image: bool):
    is_running = False
    number_ugv_patch = 1
    patches_width = 450

    if save_to_image:
        (default_path / "results").mkdir(exist_ok=True)

    results = pd.DataFrame(columns=["current node", "next node", "ugv x", "ugv y", "naive is in node",
                                    "inference duration"]
                                   + number_ugv_patch * [str(i) for i in range(keep_best)])
    anim = RobotAnimator(graph, match_count_thresh, match_count_probable_thresh)

    ugv_rgb_path = pd.DataFrame(np.array([(int(path.stem.split("_")[-1]), path)
                                          for path in load_paths(hub_drone / "ugv_rgb")]))
    ugv_rgb_bev_path = pd.DataFrame(np.array([(int(path.stem.split("_")[-1]), path)
                                              for path in load_paths(hub_drone / "ugv_rgb_bev")]))

    def closest_images(timestamp):
        return (cv2.imread(ugv_rgb_path.iloc[(ugv_rgb_path[0] - timestamp).abs().argmin()][1]),
                ugv_rgb_bev_path.iloc[(ugv_rgb_bev_path[0] - timestamp).abs().argmin()][1])

    for index, (next_node, current_node, (_, search_gnss)) in tqdm(
            enumerate(zip(next_nodes,
                          current_nodes,
                          gnss.iterrows()))):
        if index < 230:
            continue
        if next_node is None:
            break

        ugv_time = search_gnss["timestamp"]
        ugv_image, filepath = closest_images(ugv_time)
        ugv_bev = cv2.imread(filepath)
        dt = time.time_ns()
        ugv_2d_position = np.array([search_gnss['longitude'], search_gnss['latitude']])

        patches = create_matrix_front_patch(patches_width,
                                            ugv_bev.shape[:2],
                                            pattern=(1, number_ugv_patch),
                                            margin=(0, np.ceil(patches_width * (np.sqrt(2) - 1) / 2)))
        yaw_ugv = search_gnss['gps_heading'] # angle from global to ugv
        patches = apply_rotations(patches, [np.rad2deg(-yaw_ugv)])
        extracted_patches = [extract_patch(p, ugv_bev, patches_width) for p in patches]

        if viz:
            for p in patches:
                cv2.polylines(ugv_bev, [p], isClosed=True, color=(255, 255, 255), thickness=10)
                # cv2.putText(ugv_bev,
                #             f"{'-> ' + str(next_node) if current_node is None else 'at ' + str(current_node)}",
                #             np.mean(p, axis=0).astype(np.int32),
                #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7)
            cv2.imshow("bev", resize_image(ugv_bev, 800, 800))

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
                if not is_in_node and mkpts_0.shape[0] > match_count_thresh:
                    is_in_node = True
            correspondances_each_pairs.append(correspondances)
        dt = time.time_ns() - dt
        anim.update_robot_pose(*ugv_2d_position, yaw_ugv)
        anim.update_node_display(next_node, np.array([[mkpts_0.shape[0] for mkpts_0, _ in correspondances]
                                                      for correspondances in correspondances_each_pairs]).max(axis=0))

        results.loc[len(results)] = [current_node, next_node, *ugv_2d_position, is_in_node, dt / 10 ** 9,
                                     *[correspondances_each_pairs[i][j][0].shape[0] if
                                       len(correspondances_each_pairs[i]) > j else 0
                                       for j in range(10)
                                       for i in range(number_ugv_patch)]]
        if index % 100 == 0:
            results.to_csv(str(default_path / "results.csv"), index=False)

        match_grid = generate_match_grid(next_node, ugv_2d_position, extracted_patches,
                                         correspondances_each_pairs, match_count_thresh)
        if viz:
            cv2.imshow("match_grid", resize_image(match_grid, 2500, 1000))
        if save_to_image:
            cv2.imwrite(str(default_path / "results" / f"match_grid_{index}.png"), match_grid)

        patch_location = generate_patch_location(next_node, correspondances_each_pairs, match_count_thresh)
        if patch_location is not None:
            patch_location = resize_image(patch_location, 1000, 1000)
            if viz:
                cv2.imshow("patch_location", patch_location)

        location = anim.render()
        if viz:
            cv2.imshow("location", location)

        # bottom = np.hstack([location[:, :, 0:3],
        #                     resize_image(cv2.resize(ugv_image, (0, 0), fx=2.3, fy=2.3, interpolation=cv2.INTER_AREA),
        #                                  max_height=800),
        #                     resize_image(cv2.resize(ugv_bev, (0, 0), fx=1.6, fy=1.6, interpolation=cv2.INTER_AREA),
        #                                  max_height=800)])
        # presentation = np.vstack([resize_image(match_grid, bottom.shape[1]), bottom])
        # presentation = resize_image(presentation, 1920, 1080)
        # if presentation.shape[0] < 1080:
        #     bottom = 255 * np.ones((1080 - presentation.shape[0],
        #                             presentation.shape[1], 3),
        #                            dtype=np.uint8)
        #     cv2.putText(bottom,
        #                 "The top shows matching between each UGV patch (rows) and each UAV patch of the next node (columns)."
        #                 " When there are enough correspondences, a blue rectangle indicates corresponding areas.",
        #                 (0, 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        #     cv2.putText(bottom,
        #                 "The bottom left image is the topological map where the UGV evolves."
        #                 " The rectangles represents the collected UAV patches."
        #                 " They turn green when the patch is detected by the UGV.",
        #                 (0, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        #     cv2.putText(bottom,
        #                 "The bottom center image is the view of the UGV while the bottom right shows the same image deformed to obtain a BEV.",
        #                 (0, 80),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        #     cv2.putText(bottom,
        #                 "The white rectangles correspond to the UGV patches used in the matching.",
        #                 (0, 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        #     presentation = np.vstack([presentation, bottom])
        # if presentation.shape[1] < 1920:
        #     presentation = np.hstack([presentation, 255 * np.ones((presentation.shape[0],
        #                                                            1920 - presentation.shape[1], 3),
        #                                                           dtype=np.uint8)])
        # if viz:
        #     cv2.imshow("Presentation", presentation)
        # if save_to_image:
        #     cv2.imwrite(str(default_path / "results" / f"presentation_{index}.png"), presentation)

        k = cv2.waitKey(5 if is_running else 0) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            is_running = not is_running

    results.to_csv(str(default_path / "results.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_best', type=int, default=10)
    parser.add_argument('--too_close_thresh', type=float, default=0.000045)
    parser.add_argument('--match_count_thresh', type=float, default=600)
    parser.add_argument('--match_count_probable_thresh', type=float, default=250)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--only-show-results', dest='only_show_results', action='store_true')
    parser.add_argument('--scouting', dest='should_generate_scouting_data', action='store_true')
    parser.add_argument('--filter', dest='should_filter_scouting_data', action='store_true')
    parser.add_argument('--log_level', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.setLevel(getattr(logging, args.log_level.upper()))

    if args.only_show_results:
        indexes = list(set(sorted([int(f.stem
                                       .replace("match", "")
                                       .replace("bev", "")
                                       .replace("location", "")) for f in (default_path / "results").glob(f"*.png")])))

        for i in indexes:
            cv2.imshow("bev", cv2.imread(default_path / "results" / f"bev{i}.png"))
            cv2.imshow("match_grid", cv2.imread(default_path / "results" / f"match{i}.png"))
            cv2.imshow("location", cv2.imread(default_path / "results" / f"location{i}.png"))

            k = cv2.waitKey(200) & 0xFF
            if k == ord('q'):
                break
        exit(0)

    # load position of robots
    uav_gnss = pd.read_csv(get_dataset_by_name("rosbag_u2is/hub_drone_130625/uav_gnss_extended.csv"))
    uav_gnss = uav_gnss[uav_gnss["timestamp"].between(1749828491088473915, 1749828753755170238)]
    ugv_gnss = pd.read_csv(get_dataset_by_name("rosbag_u2is/hub_drone_130625/ugv_gnss_extended.csv"))

    if (default_path / "graph.csv").exists():
        graph = Graph()
        graph.load()
    else:
        # graph = create_norlab_graph()

        def draw_arrow(df, yaw_column, l, is_degree):
            for i in range(0, len(df), int(len(df) * 0.05)):
                curr_gnss = df.iloc[i]
                yaw = curr_gnss[yaw_column] / 180 * np.pi if is_degree else curr_gnss[yaw_column]
                ax.quiver(curr_gnss["longitude"], curr_gnss["latitude"],
                          l * np.cos(yaw), l * np.sin(yaw),
                          angles='xy', scale_units='xy', scale=1, color='r')

        graph = create_u2is_graph()
        fig, ax = plt.subplots()
        graph.plot(ax)
        ax.plot(uav_gnss["longitude"], uav_gnss["latitude"])
        l = 0.0001
        # draw_arrow(uav_gnss, "yaw", l, False)
        ax.plot(ugv_gnss["longitude"], ugv_gnss["latitude"])
        draw_arrow(ugv_gnss, "gps_heading", l, False)
        ax.grid()
        plt.tight_layout()
        ax.axis('equal')
        fig.show()
        graph.save()

    viz = args.viz
    if args.should_generate_scouting_data:
        generate_scouting_data(graph, uav_gnss, viz)
        graph.save()

    if args.should_filter_scouting_data:
        filter_scouting_data(graph, args.keep_best, args.too_close_thresh)
        graph.save()

    current_nodes, next_nodes = get_path_in_node(ugv_gnss, graph)
    detect_ugv_location(graph, next_nodes, current_nodes, ugv_gnss,
                        args.keep_best, args.match_count_thresh, args.match_count_probable_thresh,
                        True, True)


if __name__ == "__main__":
    main()
