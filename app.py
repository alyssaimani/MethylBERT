from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import pandas as pd
import os
import numpy as np
from Bio import SeqIO
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from models.Neo import ModelNeo
from transformers import AutoTokenizer, AutoModel
from preprocess import get_path, preprocess_data, label_data, tokenize, neo_tokenize, windowing

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'fasta'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def initialize_BERT(model_name):
#     # Load the ProtBERT tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # import ProtBERT pretrained model
#     bert_model = AutoModel.from_pretrained(model_name)
#     return tokenizer, bert_model

# tokenizer, bert_model = initialize_BERT('Rostlab/prot_bert')


@app.route('/')
def index():
    return render_template('index.html', page=request.path)


@app.route('/about')
def about():
    return render_template('about.html', page=request.path)


@app.route('/predict', methods=['POST'])
def predict():
    input_seq = request.form.get('sequence')
    if 'file_seq' not in request.files:
        flash('No file part')
        return redirect('/')
    file_seq = request.files['file_seq']
    if file_seq.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file_seq and allowed_file(file_seq.filename):
        filename = secure_filename(file_seq.filename)
        file_seq.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)

    spliced_seq, positions = windowing(input_seq)
    residues = np.array([list(residue) for residue in spliced_seq])
    tokens = neo_tokenize(residues)
    tokens = torch.from_numpy(tokens.astype(np.int)).cuda()

    model = ModelNeo().cuda()
    model.load_state_dict(torch.load("checkpoints/best_model_merged.pth"))

    with torch.no_grad():
        preds = model(tokens)
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)
    pred_classes = ["Positive" if pred == 1 else "Negative" for pred in preds]
    results = [(seq, site, pred)
               for seq, site, pred in zip(spliced_seq, positions, pred_classes)]
    # paths = get_path('data')
    # test_neg = pd.read_csv(paths[2], sep='\t', header=None)
    # test_pos = pd.read_csv(paths[8], sep='\t', header=None)

    # clean_test_neg = preprocess_data(test_neg)
    # clean_test_pos = preprocess_data(test_pos)

    # negative_seq_test = np.array([ list(word) for word in clean_test_neg.sequence.values])
    # positive_seq_test = np.array([ list(word) for word in clean_test_pos.sequence.values])

    # negative_lab_test = np.zeros((negative_seq_test.shape[0],), dtype=int)
    # positive_lab_test = np.ones((positive_seq_test.shape[0],), dtype=int)

    # dataset_X_test = np.concatenate((positive_seq_test, negative_seq_test), axis=0, out=None)
    # dataset_Y_test = np.concatenate((positive_lab_test, negative_lab_test), axis=0, out=None)

    # dataset_X_token_test = neo_tokenize(dataset_X_test)

    # X_test, y_test = shuffle(dataset_X_token_test, dataset_Y_test, random_state=13)

    # X_test_torch = torch.from_numpy(X_test.astype(np.int)).cuda()
    # y_test_torch = torch.from_numpy(y_test.astype(np.int)).cuda()
    # print(X_test_torch[0])

    # model = ModelNeo().cuda()
    # model.load_state_dict(torch.load("checkpoints/best_model_merged.pth"))

    # with torch.no_grad():
    #     y_predicted_test = model(X_test_torch)
    #     acc_test = torch.max(y_predicted_test, 1)[1].eq(
    #         y_test_torch).sum() / float(y_test_torch.shape[0])
    #     f1_score_test = f1_score(torch.max(y_predicted_test, 1)[1].cpu(
    #     ).numpy(), y_test_torch.cpu().numpy(), average='macro')
    #     mcc_test = matthews_corrcoef(torch.max(y_predicted_test, 1)[
    #                                 1].cpu().numpy(), y_test_torch.cpu().numpy())
    #     fpr, tpr, thresholds = metrics.roc_curve(y_test_torch.cpu().numpy(
    #     ), torch.max(y_predicted_test, 1)[1].cpu().numpy(), pos_label=1)
    #     auc_test = metrics.auc(fpr, tpr)

    # print(f'Test accuracy: {acc_test.item():.4f}, F1: {f1_score_test.item():.4f}, mcc: {mcc_test.item():.4f}, auc: {auc_test.item():.4f}')
    # print(confusion_matrix(torch.max(y_predicted_test, 1)
    #     [1].cpu().numpy(), y_test_torch.cpu().numpy()))

    # Preprocess for BERT
    # test_seq, test_mask = tokenize(test_x, tokenizer)
    # print(test_seq[0])

    # test_seq = torch.tensor(test_seq)
    # test_mask = torch.tensor(test_mask)
    # test_label = torch.tensor(test_y.tolist())
    # print(test_seq[0])

    # model = ModelNeo().cuda()
    # model = model.load_state_dict(torch.load("best_model.pth"))
    # with torch.no_grad():
    #     prediction = model()
    return render_template('predict.html', seq=input_seq, results=results)


if __name__ == "__main__":
    app.run(debug=True)
