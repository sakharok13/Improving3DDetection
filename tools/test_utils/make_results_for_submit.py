import os
import pickle

def format_once_submit_results():
    root_dir = "/home/junbo/ssd/repository/SemiDet3D/output/once_models/sup_models/second/second_4GPU_tta_4stage_cyc03_ep75_60.78_test/submission_20220131_2"
	# load 预测的结果
    prediction = "/home/junbo/ssd/repository/SemiDet3D/output/once_models/sup_models/second/second_4GPU_tta_4stage_cyc03_ep75_60.78_test/eval/epoch_73/test/default/result.pkl"
    with open(os.path.join(root_dir, prediction), "rb") as F:
        results = pickle.load(F)

    results_map = {}
    for idx, result in enumerate(results):
        results_map[result["frame_id"]] = result

	# load 模板
    with open(os.path.join("/home/junbo/ssd/repository/SemiDet3D/tools/test_utils/", "result_template.pkl"), "rb") as Ft:
        results_template = pickle.load(Ft)

    for idx, result in enumerate(results_template):
        results_template[idx] = results_map[result["frame_id"]]

	# 保存最终提交版本
    with open(os.path.join(root_dir, "result.pkl"), "wb") as Fo:
        pickle.dump(results_template, Fo)

if __name__ == '__main__':
    format_once_submit_results()