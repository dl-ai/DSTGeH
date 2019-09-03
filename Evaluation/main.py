import numpy as np
from get_data import *
from evaluation import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hash_file_path', type=str, help='Path for hash code file')
    parser.add_argument('hash_file', type=str, help='Name for hash code file')
    parser.add_argument('label_file_path', type=str, help='Path for label file')
    parser.add_argument('label_file', type=str, help='Name for label file')
    parser.add_argument('label_compare', type=int, help='single label or multiple label')

    parser.add_argument('-result_path_name', type=str, help='Path and Name for writing result file')

    parser.add_argument('--ln', type=int, action='append', help='[l,n,l,n ... ]->[[l,n],[l,n],...] where l is the seleted label and n is the seleted number')
    parser.add_argument('--qnum', type=int, help='number of query by randomly select')

    parser.add_argument("mAP", choices=[True, False], type=bool, help="Whether calculate mAP")
    parser.add_argument('P', choices=[True, False], type=bool, help="Whether calculate P")
    parser.add_argument('PR', choices=[True, False], type=bool, help="Whether calculate PR")

    parser.add_argument('--mAP_R', type=int, help='mAP scope')
    parser.add_argument('--P_HR', type=int, help='Hamming Radius for P')
    parser.add_argument('--PR_recall', type=float, action='append', help='recall checkpoint for PR')
    parser.add_argument('--PR_cls_num', type=int, action='append', help='each class number in hash file for PR')

    opt = parser.parse_args()

    hashs = get_test_hashcode(opt.hash_file_path, opt.hash_file)
    labels = get_test_label(opt.label_file_path, opt.label_file)

    label_style = opt.label_compare
    query = Create_query(hashs, labels)

    if label_style == 1:
        qs = np.array(opt.ln)
        qs_len = len(qs)
        qs = qs.reshape(int(qs_len/2),2)
        query_hash_list,query_label_list = query.get_certain_query_data(qs)
    if label_style == 2:
        query_hash_list, query_label_list = query.get_rand_query_data(opt.qnum)

    evaluation = Evaluation(hashs, labels, query_hash_list, query_label_list,label_style)
    with open(opt.result_path_name, 'w') as f:
        if opt.mAP:
            mAP = evaluation.get_mAP(opt.mAP_R)
            f.write("\n mAP: \n")
            f.writelines(str(mAP))
        if opt.P:
            P = evaluation.get_Precision_HR(opt.P_HR)
            f.write("\n P: \n")
            f.writelines(str(P))
        if opt.PR:
            PR = evaluation.get_PrecisionRecall(opt.PR_cls_num, opt.PR_recall)
            f.write("\n PR: \n")
            f.writelines(str(PR))


