import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
from ExecutionAttributes import ExecutionAttribute
from TimeCallback import TimeCallback
from keras import backend as K
import os


def write_summary_txt(execattr : ExecutionAttribute, network_format, image_format, labels, time_callback: TimeCallback, stopped_epoch):
    with open(execattr.curr_basename + ".txt", "a") as f:
        f.write('EXECUTION SUMMARY\n')
        f.write('-----------------\n\n')
        if execattr.architecture != "":
            f.write('Architecture: ' + execattr.architecture + '\n')
        else:
            f.write('Architecture: From scratch\n')

        f.write('Fusion: ' + execattr.fusion + '\n')
        f.write('Execution Seq: ' + str(execattr.seq) + '\n')
        f.write('Network Type: ' + network_format + '\n')
        f.write('Image Format: ' + image_format + '\n')
        f.write('Image Size: (' + str(execattr.img_width) + ',' + str(execattr.img_height) + ')\n')
        f.write('Date: ' + datetime.now().strftime("%Y%m%d") + '\n')
        f.write('Cycle Started at: ' + execattr.init_timestamp.strftime("%Y%m%d-%H%M%S") + '\n')
        f.write('Cycle Finished at: ' + datetime.now().strftime("%Y%m%d-%H%M%S") + '\n')
        
        if execattr.csv_path != "":
            f.write('CSV File: ' + execattr.csv_path + '\n')

        if execattr.numpy_path is not None:
            f.write('Numpy Path: ' + execattr.numpy_path + '\n')

        if execattr.fusion != "None":
            f.write('Train Samples: ' + str(execattr.train_samples) + '\n')
        else:
            f.write('Train Data Path: ' + execattr.train_data_dir + '\n')
            f.write('Train Samples: ' + str(len(execattr.train_generator.filenames)) + '\n')
        f.write('Train Steps: ' + str(execattr.steps_train) + '\n')

        if execattr.fusion != "None":
            f.write('Validation Samples: ' + str(execattr.valid_samples) + '\n')
        else:
            f.write('Validation Data Path: ' + execattr.validation_data_dir + '\n')
            f.write('Validation Samples: ' + str(len(execattr.validation_generator.filenames)) + '\n')
        f.write('Validation Steps: ' + str(execattr.steps_valid) + '\n')

        if execattr.fusion != "None":
            f.write('Test Samples: ' + str(execattr.test_samples) + '\n')
        else:
            f.write('Test Data Path: ' + execattr.test_data_dir + '\n')
            f.write('Test Samples: ' + str(len(execattr.test_generator.filenames)) + '\n')

        f.write('Test Steps: ' + str(execattr.steps_test) + '\n')
        f.write('Epochs: ' + str(execattr.epochs) + '\n')
        if stopped_epoch == 0:
            f.write('Stopped Epoch: ' + str(execattr.epochs) + '\n')
        else:
            f.write('Stopped Epoch: ' + str(stopped_epoch) + '\n')

        f.write('Batch Size: ' + str(execattr.batch_size) + '\n')
        f.write('Learning Rate: ' + str(K.eval(execattr.model.optimizer.lr)) + '\n')

        if execattr.fnames_test is None:
            filenames = execattr.test_generator.filenames
        else:
            filenames = execattr.fnames_test

        nb_samples = len(filenames)

        print(filenames)
        print(nb_samples)
        f.write("Test Generator Filenames:\n")
        print(filenames, file=f)
        f.write("\nNumber of Test Samples:\n")
        f.write(str(nb_samples) + "\n\n")

        score_gen = execattr.model.evaluate_generator(generator=execattr.test_generator, workers=1, use_multiprocessing=False, steps=execattr.steps_test, verbose=1)

        print(score)
        print('Test Loss:', score_gen[0])
        print('Test accuracy:', score_gen[1])
        f.write('Test Loss:' + str(score_gen[0]) + '\n')
        f.write('Test accuracy:' + str(score_gen[1]) + '\n\n')

        print("Training Epochs Duration: ")
        print(time_callback.times)

        f.write("\n\nTraining Epochs Duration: " + "\n")
        print(time_callback.times, file=f)

        # Confusion Matrix and Classification Report
        if execattr.fusion == "None":
            execattr.test_generator.reset()

        if execattr.architecture != "":
            Y_pred = execattr.model.predict_generator(execattr.test_generator, workers=1, use_multiprocessing=False, steps=execattr.steps_test, verbose=1)
            y_pred = np.argmax(Y_pred, axis=1)

            print(Y_pred)
            print(y_pred)

            if execattr.fusion != "None":
                classes = execattr.labels_test
            else:
                classes = execattr.test_generator.classes

            print(classes)

            f.write('Predicted Values: \n')
            print(Y_pred, file=f)
            f.write('\nRounded Values: \n')
            print(y_pred, file=f)
            f.write('\nClasses: \n')
            print(classes, file=f)

            mtx = confusion_matrix(classes, y_pred)
            print('Confusion Matrix:')
            print(mtx)
            f.write('\n\nConfusion Matrix:\n')
            f.write('TP   FP\n')
            f.write('FN   TN\n')
            print(mtx, file=f)

            plt.imshow(mtx, cmap='binary', interpolation='None')
            plt.savefig(execattr.curr_basename + '-confusion_matrix.png')
            plt.clf()

            print('Classification Report')
            print('Classification Report:', file=f)
            if execattr.fusion != "None":
                target_names = labels
            else:
                target_names = list(execattr.test_generator.class_indices.keys())
            print(classification_report(classes, y_pred, target_names=target_names))
            print(classification_report(classes, y_pred, target_names=target_names), file=f)

            cohen_score = cohen_kappa_score(classes, y_pred)
            print("Kappa Score = " + str(cohen_score))
            f.write("Kappa Score = " + str(cohen_score))

            auc_score = roc_auc_score(classes, y_pred)
            print("ROC AUC Score = " + str(auc_score))
            f.write("\n\nROC AUC Score = " + str(auc_score))

            # plot the roc curve for the model
            fpr, tpr, thresholds = roc_curve(classes, y_pred)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.savefig(execattr.curr_basename + '-roc-curve.png')
            plt.clf()

        else:    
            Y_pred = execattr.model.predict_generator(execattr.test_generator, workers=1, use_multiprocessing=False, steps=execattr.steps_test, verbose=1)
            y_pred = np.rint(Y_pred)

            print(Y_pred)
            print(y_pred)

            if execattr.fusion != "None":
                classes = execattr.labels_test
            else:
                classes = execattr.test_generator.classes

            print(classes)

            f.write('Predicted Values: \n')
            print(Y_pred, file=f)
            f.write('\nRounded Values: \n')
            print(y_pred, file=f)
            f.write('\nClasses: \n')
            print(classes, file=f)

            mtx = confusion_matrix(classes, y_pred, labels=[1, 0])
            print('Confusion Matrix:')
            print(mtx)
            f.write('\n\nConfusion Matrix:\n')
            f.write('TP   FP\n')
            f.write('FN   TN\n')
            print(mtx, file=f)

            plt.imshow(mtx, cmap='binary', interpolation='None')
            plt.savefig(execattr.curr_basename + '-confusion_matrix.png')
            plt.clf()

            # print('Classification Report')
            target_names = labels
            print(classification_report(classes, y_pred, target_names=target_names))
            print(classification_report(classes, y_pred, target_names=target_names), file=f)

            cohen_score = cohen_kappa_score(classes, y_pred)
            print("Kappa Score = " + str(cohen_score))
            f.write("Kappa Score = " + str(cohen_score))

            auc_score = roc_auc_score(classes, y_pred)
            print("ROC AUC Score = " + str(auc_score))
            f.write("\n\nROC AUC Score = " + str(auc_score))

            # plot the roc curve for the model
            fpr, tpr, thresholds = roc_curve(classes, y_pred)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(fpr, tpr, marker='.')
            plt.savefig(execattr.curr_basename + '-roc-curve.png')
            plt.clf()

        f.close()

    write_csv_test_result(execattr, score_gen, y_pred, mtx, classes, stopped_epoch)


def plot_train_stats(history, filename_loss, filename_accuracy):
    # plot history
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')

    ## As loss always exists
    pepochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss

    plt.figure(1)
    for l in loss_list:
        plt.plot(pepochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(pepochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename_loss)
    plt.clf()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(pepochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(pepochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename_accuracy)
    plt.clf()


def write_csv_test_result(execattr: ExecutionAttribute, score_gen, y_pred, mtx, classes, stopped_epoch):
    with open(execattr.summ_basename + ".csv", "a") as csv:
        if execattr.seq == 1:
            csv.write('Test Loss, Test Accuracy, TP, FP, TN, FN, Precision, Recall, F-Score, Support, Stopped Epoch\n')

        precision, recall, fscore, support = score(classes, y_pred, average='macro')

        csv.write(str(score_gen[0]) + ',' + str(score_gen[1]) + ',' + str(mtx[0, 0]) + ',' + str(mtx[1, 0]) + ',' + str(
            mtx[1, 1]) + ',' + str(mtx[0, 1]) + ',' + str(precision) + ',' + str(recall) + ',' + str(
            fscore) + ',' + str(support) + ',' + str(stopped_epoch) + '\n')

        csv.close()


def create_results_dir(basepath, network_format, image_format):
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    if not os.path.exists(basepath + '/' + network_format):
        os.makedirs(basepath + '/' + network_format)

    if not os.path.exists(basepath + '/' + network_format + '/' + image_format):
        os.makedirs(basepath + '/' + network_format + '/' + image_format)

    return basepath + '/' + network_format + '/' + image_format


def get_base_name(basepath):
    return basepath + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")


def save_model(execattr: ExecutionAttribute):
    execattr.model.save(execattr.curr_basename + "-model.h5")


def copy_to_s3(execattr: ExecutionAttribute):
    src_dir = execattr.summ_basename.split('/')
    os.system("aws s3 cp "+execattr.summ_basename.replace(src_dir[len(src_dir)-1],"") + " s3://pyliver-logs/logs/ --recursive --exclude \"*\" --include \"*" + src_dir[len(src_dir)-1] + "*\"")
