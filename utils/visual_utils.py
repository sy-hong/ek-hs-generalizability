# file manipulation, stats
import os 
import numpy as np

'''
Process current epoch's results for visualization/error analysis
> args: args from main
> mode: train/dev/test
> epoch_num: current epoch
> texts: texts of data sample
> all_preds: predictions
> all_golds: true labels
'''
def pred_types_summarization(args, mode, epoch_num, texts, all_preds, all_golds):
    
    # file paths for storing 1) correct predictions, 2) missed predictions, and 3) per-epoch summary
    correct_preds = "stat/"+ args.train_data_name + "/" + args.test_data_name + "/" + args.emo_num + "/correct_preds.csv"
    missed_preds = "stat/"+ args.train_data_name + "/" + args.test_data_name + "/" + args.emo_num +"/missed_preds.csv"
    summary = "stat/"+ args.train_data_name + "/" + args.test_data_name + "/" + args.emo_num + "/summary.csv"
    
    # correct & missed predictions
    correct = []
    missed = []

    # collect correct & missed predictions
    total_cnt = len(all_preds)
    for ind, i in enumerate(all_golds):
        if np.all(all_golds[ind] == all_preds[ind]):
            correct.append(ind)

        if np.any(all_golds[ind] != all_preds[ind]): # and (ind not in no_pred)
            missed.append(ind)

    # store predictions -- [text, pred] -- to file for visualization
    pred_write_to_file(missed_preds, mode, epoch_num, texts, missed, all_preds, all_golds)
    pred_write_to_file(correct_preds, mode, epoch_num, texts, correct, all_preds, all_golds)
 
    exp_vis(args, mode, epoch_num, total_cnt, summary, missed_preds)
    exp_vis(args, mode, epoch_num, total_cnt, summary, correct_preds)


'''
See predictions
> file path: path to save file
> mode: train/dev/test
> epoch_num: current epoch
> texts: texts of data sample
> pred_type_list: predictions
> all_preds: predictions
> all_golds: true labels
'''
def pred_write_to_file(file_path, mode, epoch_num, texts, pred_type_list, all_preds, all_golds):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(os.path.join(file_path), 'wt+', newline='', encoding="utf-8") as out_file:
        out_file.writelines("Mode: " + mode)
        out_file.writelines("\n")
        out_file.writelines("#####################################################")
        out_file.writelines("\n")
        out_file.writelines("Epoch: " + str(epoch_num))
        out_file.writelines("\n")
        out_file.writelines("#####################################################")
        out_file.writelines("\n")
        out_file.writelines("text, pred, true label")
        out_file.writelines("\n")

        for tx in pred_type_list:
            line = texts[tx] + ', ' + str(all_preds[tx]) + ', ' + str(all_golds[tx])
            out_file.writelines(line)
            out_file.writelines("\n")

        out_file.writelines("\n")   
        out_file.close()

'''
See prediction summary 
> args: args from main
> mode: train/dev/test
> epoch_num: current epoch
> total_cnt: total samples count
> op_file: output file
> rd_file: source file
'''
def exp_vis(args, mode, epoch_num, total_cnt, op_file, rd_file):
    cnt = 0   
    with open(os.path.join(op_file), 'at+', newline='', encoding="utf-8") as out_file:
        with open(os.path.join(rd_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in lines:
                cnt += 1
         
            out_file.writelines('Statistics of ' + rd_file) 
            out_file.writelines("\n")
            out_file.writelines("#####################################################")
            out_file.writelines("\n")
            out_file.writelines("Epoch: " + str(epoch_num)) 
            out_file.writelines("\n")
            out_file.writelines("#####################################################")
            out_file.writelines("\n")
            out_file.writelines("Model: " + args.model)
            out_file.writelines("\n")
            out_file.writelines("Mode: " + mode) 
            out_file.writelines("\n")
            out_file.writelines("Main dataset: " + args.main_dataset)
            out_file.writelines("\n")
            out_file.writelines("Save path: " + args.save_path)
            out_file.writelines("\n")
            out_file.writelines("Log file: " + args.logfile)
            out_file.writelines("\n")
            out_file.writelines("Prediction outcome statistics")
            out_file.writelines("\n")
            out_file.writelines("Text count: " + str(cnt))
            out_file.writelines("\n")
            out_file.writelines("Percentage: " + str((cnt/total_cnt)*100))
            out_file.writelines("\n")
            out_file.writelines("\n")
                
