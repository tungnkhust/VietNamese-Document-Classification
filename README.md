# Vietnamese Document classification

Phân loại văn bản là một bài toán điển hình trong lĩnh vực xử lý ngôn ngữ tự nhiên.
Đây là một bài toán hay với độ phức tạp khác nhau phụ thuộc vào từng chủ đề riêng của bài toán và loại văn bản cần phân loại.
Hiện tại đã có rất nhiều hướng tiếp cận như sử dụng các giải thuật học máy truyền thống (Naive Bayes, SVM) trong phân loại hay các kĩ thuật hiện đại hơn như học sâu đều cho những kết quả rất khả quan.
Trong project này tôi đã tiếp cận theo cả 2 hướng. Với hướng tiếp cận theo học máy truyền thông tôi sử dụng kỹ thuật TF-IDF để trích xuất đặc trưng và giải thuật SVM để phân loại.
Với hướng tiếp cận thứ 2 tôi đã thử nghiệm mô hình Bert với pre-trained PhoBert. Kết quả sử dụng cả 2 phương pháp đều rất khả quan, tuy nhiên thật bất ngờ việc sử dụng giải thuật SVM lại cho kết quả tốt hơn.

## Dữ liệu
Dữ liệu tôi sử dụng là bộ [VNTC](https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics
) gồm 10 chủ đề khác nhau.

## Cài đặt môi trường
Nếu bạn không có môi trường ảo anaconda, bạn có thể cài đặt theo hưỡng dẫn sau:
- [Install Anaconda3 on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

Sau đó bạn tạo môi trường ảo anaconda mới với tên "pj2":<br>
`conda create --name pj2 python=3.7`

Bạn kích hoạt môi trường mới:<br>
`conda activate pj2`

Cuối cùng bạn cài các thư viện cần thiết để chạy Project
```
pip install -r requirements.txt
```

## Kết quả

Kết quả sử dụng giải thuật SVM:

|Accuracy | F1 Score| Precision | Recall|
|--- | ---| ---| ---|
|92.62 |0.9018 |0.9084 |0.8973 |

Kết quả sử dụng model Bert với pre-trained PhoBert

|Accuracy | F1 Score|
|--- | ---|
|91.01 |0.8771|

Chi tiết về kết quả thực nghiệm [xem tại đây](https://docs.google.com/document/d/1FMTxcXtL3WpKLVQ3lzIRQ5nX_dVXDlzS7a6omW6E93I/edit?usp=sharing)

## Chạy thử nghiệm
Xử lý dữ liệu:
```
python preprocessing.py --train_dir data/Reuter10/train \
--test_dir data/Reuter10/test \
--cutoff 26 \
--hier True
```

train_dir và test_dir là đường dẫn đến thư mục chưa dữ liệu trên và test mà khi tải dư liệu về lưu tại đó.
cutoff là thông số để loại bỏ các từ xuất hiện không quá n lần, mặc định cutoff=0.
hier là biến để xem có xử lý dữ liệu cho mô hình học phân cấp hay không, mặc định là False
Có thể chạy mặc định với lệnh sau:

```
python preprocessing.py
```
Huấn luyện và đánh giá model:
```
python train.py --train_file data/full_data/data.csv \
--test_file data/full_data/test.csv \
--vocab_file vocab/vocab.csv \
--stopword_file vocab/stopword.txt \
--sublinear_tf True \
--kernel 'linear' \
--C 1 \
--hier True \
--seed 1337 \
--show_cm_matrix True
```
vocab_file là danh sách các từ dùng làm từ điển, mặc định là "".
stopword_file là danh sách các từ stopword, mặc định là "". 
kernel là kernel dùng trong giải thuật svm, nếu kernel là linear thì mô hình sử dụng LinearSCV() nếu không sẽ sử dụng SVC() với kernel tương ứng, mặc định là "linear".
C là thông số điều chỉnh độ chịu lỗi của giải thuật svm, mặc định là 1.
hier=True thì dùng mô hình phân cấp, mặc định là False.
show_cm_matrix=True thì sẽ hiện thị 2 confusion matrix (nomalize, non-nomalize) sau khi chạy xong kết quả lưu trong thư mục results, mặc định là False.
Chạy mặc định với lệnh sau:
```
python train.py
```

Đới với phương pháp sử dụng mô hình Bert tôi chạy trên colab, do vậy bạn cần tinh chỉnh lại cho phù hợp.

