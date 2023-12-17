# native python libraries
import argparse
from collections import defaultdict
from datetime import datetime
import logging
import os
import pickle

# ML, stats
import numpy as np
import sklearn.metrics as skmetric

# pytorch
import torch
import torch.nn as nn

# transformers
from transformers import AdamW
import transformers.optimization as tfoptim

# helpers
import configs.defaults as configs
import utils.batch_utils as butils
import utils.data_utils as dutils
import utils.loss_utils as lutils
import utils.meta_utils as meutils
import utils.model_utils as moutils
import utils.visual_utils as visutils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE', DEVICE)
SEED = configs.SEEDS[0]
torch.cuda.empty_cache()

##################################################################################
# TRAIN, EVALUATE, PREDICT
##################################################################################
"""
Run the given model on the given data and return its predictions
> model: an instance of one of our models (models/)
> data_loader: an instance of one of our batch generators (utils/batch_utils)
@return: all_preds, a list of predictions (for one task)
"""
def predict(model, data_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, _) in enumerate(data_loader.get_batches()):
            # get model predictions
            preds = model.predict(dataset_id, tokens, token_types, attn_mask)

            all_preds.extend(preds.tolist())

    return all_preds

"""
Evaluate the model on the given data and return its scores
> args: main args
> mode: eval
> epoch: current epoch
> model: an instance of one of our models (models/)
> tokenizer: tokenizer
> data_loader: an instance of one of our batch generators (utils/batch_utils)
> metrics: a dictionary of callable metric functions that operate on the output of our models
@return: calculated_metrics, a dictionary with the same keys as metrics but calculated numbers as the values
"""
def evaluate(args, mode, epoch, model, tokenizer, data_loader, metrics):
    model.eval()
    all_golds = []
    all_preds = []
    tokens_id = []

    with torch.no_grad():
        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, golds) in enumerate(data_loader.get_batches()):
            # get model predictions
            preds = model.predict(dataset_id, tokens, token_types, attn_mask)

            tokens_id.extend(tokens)
            all_golds.extend(golds.tolist())
            all_preds.extend(preds.tolist())

    texts = []

    for k in (tokens_id):
        t = tokenizer.decode(k, True)
        text = '\"' + t + '\"'
        texts.append(text)
    
    visutils.pred_types_summarization(args, mode, epoch, texts, all_preds, all_golds)

    calculated_metrics = {}
    for metric_name in metrics:
        calculated_metrics[metric_name] = metrics[metric_name](all_golds, all_preds)

    return calculated_metrics


"""
Train the given model on the given data and return the trained model and its dev/test performance
Includes early stopping
> model: an instance of one of our models (models/)
> tokenizer: tokenizer
> train_loader: an instance of one of our data loaders (utils/data_utils)
> loss_calculator: an instacne of one of our loss calculators (utils/loss_utils)
> optimizer: a Transformers optimizer
> scheduler: a Transformers scheduler
> dev_loader: an instance of one of our data loaders (utils/data_utils)
> dev_metrics: a dictionary of callable metric functions that operate on the output of our models
> target_metric_name: one of the keys from dev_metrics that will be the optimization criterion
> kwargs: kwargs generated from this file's __main__
"""
def train(model, tokenizer, train_loader, loss_calculator, optimizer, scheduler, dev_loader, dev_metrics, target_metric_name,
          kwargs):

    dev_scores = []
    best_dev_score = -np.inf
    best_epoch = 0

    # for early stopping
    epochs_wo_improvement = 0

    for i in range(kwargs.epochs):
        logging.info("=====EPOCH {i}=====".format(i=i))
        model.train()

        # run through all batches in train generator
        for j, (dataset_id, tokens, token_types, attn_mask, golds) in enumerate(train_loader.get_batches()):
            # get model predictions
            preds, loss = model.get_loss(dataset_id, tokens, token_types, attn_mask, loss_calculator, golds)

            # backward pass, clip gradients, optimizer step
            optimizer.zero_grad()
            loss.backward()
            if kwargs.clip_value > 0:
                nn.utils.clip_grad_norm_(model.parameters(), kwargs.clip_value)
            optimizer.step()
            scheduler.step()

            # log info
            if j % kwargs.print_every == 0:
                logging.info("[Epoch {i}] [Batch {j}/{batches}] loss={loss:.4f}".format(
                    i=i, j=j, batches=len(train_loader), loss=loss
                ))

        # get dev performance
        dev_results = evaluate(kwargs, "TRAIN", i, model, tokenizer, dev_loader, dev_metrics)

        logging.info("DEV RESULTS FOR EPOCH {i}:".format(i=i))
        for metric in dev_results:
            logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                    metric_value=dev_results[metric]))

        # need to check if no improvement >= tolerance for patience epochs
        # save parameters with the best epoch
        dev_scores.append(dev_results[target_metric_name])

        # if this is the strictly best epoch (or the first epoch), save its params
        # note: this "best score" check bypasses the tolerance requirement
        if len(dev_scores) == 1 or dev_scores[-1] > max(dev_scores[:-1]):
            best_epoch = i
            best_dev_score = dev_scores[-1]
            epochs_wo_improvement = 0
            torch.save(model.state_dict(), kwargs.tmp_fn)
        elif dev_scores[-1] >= dev_scores[-2] + kwargs.tolerance:
            # otherwise, if at least improving, reset the patience
            epochs_wo_improvement = 0
        else:
            # if not improving, increment the patience
            epochs_wo_improvement += 1
            logging.info("No improvement for {j} epoch{s}".format(j=epochs_wo_improvement,
                                                                      s="" if epochs_wo_improvement == 1 else "s"))
            # if we have reached the patience, stop training
            if epochs_wo_improvement >= kwargs.patience:
                logging.info("Patience exceeded at epoch {i}".format(i=i))
                break

    # before stopping, load the best set of parameters if they were not the last set
    # then delete the temp file
    if best_epoch != kwargs.epochs - 1:
        model.load_state_dict(torch.load(kwargs.tmp_fn))

    if os.path.exists(kwargs.tmp_fn):
        os.remove(kwargs.tmp_fn)

    logging.info("Best epoch was {j}, "
                 "with best {metric_name} {metric_value:.4f}".format(j=best_epoch,
                                                                     metric_name=target_metric_name,
                                                                     metric_value=best_dev_score))
            
##################################################################################
# MAIN TRAIN FUNCTION
##################################################################################
"""
Train one model from scratch, including creating it and all its optimizers, loss functions, etc.
> seed: an integer random seed
> args: args in main
"""
def train_main(seed, args):
    
    # random filename for saving params -- get BEFORE setting random seed
    if args.tmp_fn is None:
        args.tmp_fn = "params-{n}.tmp".format(n=np.random.randint(10000000, 99999999))

    logging.info("Saving temporary parameters to {fn}".format(fn=args.tmp_fn))

    # set real random seed upon starting training
    meutils.set_random_seed(seed)

    logging.info("%" * 40)
    logging.info("NEW TRAINING RUN")
    logging.info("Random seed: {seed}".format(seed=seed))
    logging.info(args)
    logging.info("%" * 40)

    # for the multitask model
    simultaneous_multi = False

    logging.info("Loading data")
    all_datasets = [args.main_dataset] if args.aux_datasets is None else [args.main_dataset] + args.aux_datasets

    # create data and batcher
    # _datas will be a list of lists of tuples
    # one outer list for the datasets
    # one inner list for the tuples in that dataset
    # inside the inner list, each element is a tuple of (token_ids, type_ids, attention_mask, label1, label2, ...)
    # labels2idxes is a list of lists of dictionaries (outer list=datasets, inner list=tasks, dict=labels)
    train_datas, tokenizer, labels2idxes = dutils.get_datasets(all_datasets)
    dev_data, _, _ = dutils.get_datasets([args.dev_file], tokenizer=tokenizer, label2idx=[labels2idxes[0]])
    test_data, _, _ = dutils.get_datasets([args.test_file], tokenizer=tokenizer, label2idx=[labels2idxes[0]])

    # a lists of lists of booleans
    # one outer list the datasets, one inner list for the tasks in that dataset
    # most commonly the inner lists will be singletons
    train_is_multilabel = dutils.sniff_multilabel(train_datas)

    args.tokenizer = tokenizer

    # create model
    logging.info("Creating model")
    model, task_setting = moutils.create_model(labels2idxes, train_is_multilabel, args)
    model = model.to(DEVICE)

    # create dataloaders (can be useful for multitask)
    if simultaneous_multi:
        train_loader = butils.SimultaneousBatchGenerator(train_datas[0], args.batch_size, DEVICE,
                                                         train_is_multilabel[0], [len(l2i) for l2i in labels2idxes[0]])
    else:
        if task_setting == "single":
            # we care only about one dataset and one task
            train_loader = butils.SimpleBatchGenerator(train_datas, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                                       len(labels2idxes[0][0]))
        else:
            # we may care about multiple datasets but we care about only one task
            train_loader = butils.RoundRobinBatchGenerator(train_datas, args.batch_size, DEVICE, [tm[0] for tm in
                                                                                                  train_is_multilabel],
                                                           [len(l2i[0]) for l2i in labels2idxes])

    dev_loader = butils.SimpleBatchGenerator(dev_data, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                             len(labels2idxes[0][0]), shuffle=False)
    test_loader = butils.SimpleBatchGenerator(test_data, args.batch_size, DEVICE, train_is_multilabel[0][0],
                                             len(labels2idxes[0][0]), shuffle=False)

    # create loss functions
    if simultaneous_multi:
        loss_functions = lutils.get_simultaneous_loss_functions(train_datas, train_is_multilabel[0], DEVICE, args)
        loss_calculator = lutils.SimultaneousLossCalculator(loss_functions, args.stress_weight)
    else:
        loss_functions = lutils.get_loss_functions(train_datas, [tm[0] for tm in train_is_multilabel], DEVICE, args)
        loss_calculator = lutils.LossCalculator(loss_functions)

    # create optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # create scheduler for optimizer
    scheduler = tfoptim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        (len(train_loader) * args.epochs) / 10), num_training_steps=len(train_loader) * args.epochs)

    # create evaluation metrics
    dev_metrics = {
        "dev accuracy": lambda gold, pred: skmetric.accuracy_score(gold, pred),
        "dev f1": lambda gold, pred: skmetric.f1_score(gold, pred,
                                                       average="binary" if len(labels2idxes[0]) == 2 else "macro")
    }

    eval_metrics = {
        "eval accuracy": lambda gold, pred: skmetric.accuracy_score(gold, pred),
        "eval f1": lambda gold, pred: skmetric.f1_score(gold, pred,
                                                        average="binary" if len(labels2idxes[0]) == 2 else "macro")
    }

    # want hamming loss to be minimized
    if train_is_multilabel[0][0]:
        dev_metrics["dev hamming"] = lambda gold, pred: skmetric.hamming_loss(gold, pred)
        eval_metrics["eval hamming"] = lambda gold, pred: skmetric.hamming_loss(gold, pred)

    target_metric = "dev f1"

    # train
    logging.info("Training model")
    train(model, tokenizer, train_loader, loss_calculator, optimizer, scheduler, dev_loader, dev_metrics, target_metric, args)

    logging.info("Collecting final dev metrics for serialization")
    dev_results = evaluate(args, "FINAL DEV", 0, model, tokenizer, dev_loader, dev_metrics)

    # final evaluations
    logging.info("FINAL DEV RESULTS:")
    for metric in dev_results:
        logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                metric_value=dev_results[metric]))

    logging.info("Evaluating model on test set")
    eval_results = evaluate(args, "FINAL TEST", 0, model, tokenizer, test_loader, eval_metrics)

    logging.info("EVAL RESULTS:")
    for metric in eval_results:
        logging.info("{metric_name}: {metric_value:.4f}".format(metric_name=metric,
                                                                metric_value=eval_results[metric]))

    # return model, results, and meta-info
    meta = {"tokenizer": tokenizer,
            "labels2idxes": labels2idxes,
            "is_multilabel": train_is_multilabel,
            "args": args,
            "dev_results": dev_results,
            "eval_results": eval_results}

    return model, meta


"""
Run training in the requested way, either running 1+ random restarts or parameter optimization
> kwargs: kwargs generated from thie file's __main__
"""
def train_setup(kwargs):    

    # run model a given number of times and report average performance
    if kwargs.num_restarts == 1:
        model, meta = train_main(SEED, kwargs)

        # save model and all results if requested
        if kwargs.save_path is not None:
            logging.info("Saving entire model and meta")
            torch.save(model, kwargs.save_path + "-params.pth")
            with open(kwargs.save_path + "-meta.pkl", "wb+") as f:
                pickle.dump(meta, f)
    else:
        # run the model n times and report average results
        all_dev_results = defaultdict(list)
        all_eval_results = defaultdict(list)
        all_models = []
        all_metas = []

        for i in range(kwargs.num_restarts):
            logging.info("--------------TRAINING #{i}--------------".format(i=i))

            this_model, this_meta = train_main(configs.SEEDS[i], kwargs)

            # add metrics together
            all_models.append(this_model)
            all_metas.append(this_meta)

            for key, collector in zip(["dev_results", "eval_results"], [all_dev_results, all_eval_results]):
                for metric in this_meta[key]:
                    collector[metric].append(this_meta[key][metric])

        stdev_dev = {}
        stdev_eval = {}

        avg_dev = {}
        avg_eval = {}

        for collector, avg, stdev in zip([all_dev_results, all_eval_results],
                                            [avg_dev, avg_eval],
                                            [stdev_dev, stdev_eval]):
            for metric in collector:
                avg[metric] = np.mean(collector[metric])
                stdev[metric] = np.std(collector[metric])

        final_meta = {"avg_dev_scores": avg_dev,
                        "stdev_dev_scores": stdev_dev,
                        "avg_eval_scores": avg_eval,
                        "stdev_eval_scores": stdev_eval,
                        "model_metas": all_metas}

        if kwargs.save_path is not None:
            logging.info("Saving all models and meta")
            for i, model in enumerate(all_models):
                torch.save(model, kwargs.save_path + "-" + str(i) + "-params.pth")
            with open(kwargs.save_path + "-meta.pkl", "wb+") as f:
                pickle.dump(final_meta, f)



##################################################################################
# MAIN CONSOLE
##################################################################################
if __name__ == "__main__":
    # track time
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="actions to perform")

    ########## TRAIN ARGUMENTS ##########
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train_setup, mode="train")

    data_group = train_parser.add_argument_group("data")
    data_group.add_argument("--main_dataset", type=str, required=True,
                            help="path to the main dataset. expected to be a .csv with headers: text, label.")
    data_group.add_argument("--aux_datasets", type=str, nargs="+",
                            help="space-separated list of paths to auxiliary datasets. "
                                 "expected to be same format as main.")
    data_group.add_argument("--dev_file", type=str, required=True,
                            help="path to the dev data. expected to be same format as main.")
    data_group.add_argument("--test_file", type=str, required=True,
                            help="path to the test data. expected to be same format as main.")
    data_group.add_argument("--train_data_name", type=str, required=True,
                            help="dataset name for training")
    data_group.add_argument("--test_data_name", type=str, required=True,
                            help="dataset name for testing")
    data_group.add_argument("--emo_num", type=str, required=False,
                            help="number of emotions supplied in the auxiliary data")
    
    model_group = train_parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, choices=["baseline", "multitask"],
                             required=True,
                             help="the name of the model to use. some models may use different model args than others.")
    model_group.add_argument("--encoder", type=str, choices=["lstm", "gru"], default=configs.DEFAULTS["encoder"],
                             help="the type of recurrent layer to use for the RNN model. ignored for others.")
    model_group.add_argument("--embed_dim", type=int, default=configs.DEFAULTS["embed_dim"],
                             help="size of the embeddings used by the model. ignored for BERT.")
    model_group.add_argument("--hidden_dim", type=int, default=configs.DEFAULTS["hidden_dim"],
                             help="size of the model's hidden layer. ignored for BERT.")
    model_group.add_argument("--num_layers", type=int, default=configs.DEFAULTS["num_layers"],
                             help="number of RNN of transformer layers to use. ignored for BERT.")
    model_group.add_argument("--dropout", type=float, default=configs.DEFAULTS["dropout"],
                             help="dropout to apply to the model during training")
    model_group.add_argument("--plm", type=str, default=configs.DEFAULTS["bert"],
                             help="a path to a pretrained BERT model, or the name of such a model supported by "
                                  "huggingface (e.g., default: 'bert-base-uncased')")
    model_group.add_argument("--eta", type=bool, default=False,
                            help="train a model based on a saved model")
    model_group.add_argument("--vis", type=bool, default=False,
                            help="emo or binary")
    model_group.add_argument("--load_model_path", type=str, required=False, 
                            help="train a model based on a saved model")

    train_group = train_parser.add_argument_group("train")
    train_group.add_argument("--epochs", type=int, default=configs.DEFAULTS["epochs"],
                             help="number of epochs to train (max epochs, if early stopping).")
    train_group.add_argument("--patience", type=int, default=configs.DEFAULTS["patience"],
                             help="number of epochs to wait for validation improvement before exiting.")
    train_group.add_argument("--tolerance", type=float, default=configs.DEFAULTS["tolerance"],
                             help="amount the validation performance must improve to be considered \"improving\".")
    train_group.add_argument("--no_early_stop", action="store_true",
                             help="include this flag to turn off early stopping.")
    train_group.add_argument("--main_only_epochs", type=int, default=configs.DEFAULTS["main_only_epochs"],
                             help="number of epochs to train on only the main dataset after the initial training. "
                                  "if 0, will not train further.")
    train_group.add_argument("--batch_size", type=int, default=configs.DEFAULTS["batch_size"],
                             help="the size of the batches to use when training and evaluating.")
    train_group.add_argument("--lr", type=float, default=configs.DEFAULTS["lr"],
                             help="initial learning rate for the optimizer.")
    train_group.add_argument("--clip_value", type=float, default=configs.DEFAULTS["clip_value"],
                             help="the value at which to clip gradients. -1 for no gradient clipping.")
    train_group.add_argument("--class_weights", action="store_true",
                             help="include this flag to use class weights in the loss calculations.")
    train_group.add_argument("--stress_weight", type=float, default=configs.DEFAULTS["stress_weight"],
                             help="weight of the stress task for the Multi model. emotion is weighted with 1 - this.")

    meta_group = train_parser.add_argument_group("meta")
    meta_group.add_argument("--tmp_fn", help="an ID for the para meter checkpoints. will be randomly generated on the "
                                             "order of 10^7 if not given. checkpoints will be temporarily saved as "
                                             "'params-{tmp_fn}.tmp'.")
    meta_group.add_argument("--optimize", action="store_true",
                            help="include this flag to use the ax library to tune parameters.")
    meta_group.add_argument("--trials", type=int, default=configs.DEFAULTS["trials"],
                            help="number of hyperparameter trials to attempt. ignored if not optimizing.")
    meta_group.add_argument("--num_restarts", type=int, default=configs.DEFAULTS["num_restarts"],
                            help="the number of random restarts to average. if optimizing, run each parameter setting "
                                 "this many times. we have 10 random seeds predefined in configs/const.py; more "
                                 "restarts than this will cause an error unless you add more seeds.")
    meta_group.add_argument("--save_path", type=str, default=configs.DEFAULTS["save_path"],
                            help="the path to save parameters and metadata. we will append -params.pth or -meta.pkl to "
                                 "this string to save the data.")
    meta_group.add_argument("--print_every", type=int, default=configs.DEFAULTS["print_every"],
                            help="log training info every so many batches.")
    meta_group.add_argument("--log", action="store_true")
    meta_group.add_argument("--logfile", type=str,
                            default="{currenttime}.log".format(currenttime=datetime.now().strftime("%m%d_%H:%M:%S")))


    # parse arguments
    kwargs = parser.parse_args()

    # error checking -- throws errors if anything is wrong
    meutils.validate_args(kwargs)
    logging.info("Successfully validated arguments!")

    # set up logger
    logFormatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %I:%M:%S %p")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.pop()  

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if kwargs.log:
        fileHandler = logging.FileHandler(kwargs.logfile)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

    # run main -- predict, evaluate, or train
    kwargs.func(kwargs)

    # log total running time
    end_time = datetime.now()
    logger.info(">> Process took {t}.".format(t=str(end_time - start_time)))
