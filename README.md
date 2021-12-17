# VK_VAD_test

## Файлы 
model.py - реализован Frequential Attention + сама нейронная сеть
utils.py - утилиты (датасеты под обучение training loop'ы, аугоментации и тд)
train.py - запуск обуечения
predict.py - запуск инференса

## Инструкция по запуску обучения
Предваритльно должны быть установлены библиотеки librosa, pytorch, torchaudio, sklearn, random, numpy, tqdm.
Разархивировать датасеты (LibriSpeech , musan (https://www.openslr.org/17/), test_evaluationlib (тестовый датасет for_devs)) и оставить их с дефолтными именами 

python train.py -lbs LibriSpeech_directory -m musan_directory -n n_epoch -d device -bs batch_size 

LibriSpeech_directory - путь до директории с папкой (датасетом) LibriSpeech 
musan_directory - путь до директории с папкой (датасетом) musan 
device - cpu/cuda 
n_epoch - число эпох прогона обучения 
batch_size - размер батча 

python predict.py -tst test_evaluationlib_directory -d device -bs batch_size

test_evaluationlib_directory - путь до директории с папкой (датасетом) test_evaluationlib 
device - cpu/cuda 
batch_size - размер батча 

## Обзор существующих методов
За последние года (2020-2021) для задачи VAD в основном в работах используют архитектуры основанные на рекурентных и сверточных сетях. \
(1) https://arxiv.org/pdf/2103.03529.pdf - используют кобинацию CNN + BiLSTM, в качестве акустических признаков испльзуют log mel filterbanks. Также провели иследование зависимости качества в зависимости от типа LSTM (undiretional / bidirectional). Но к сожалению, BiLSTM не дает возможности использовать даную архитектуру в рилтайме (о чем и упоминают авторы). \
(2) https://arxiv.org/pdf/2010.13886.pdf - предложили свою сверточную архитектуру MarbleNet (основано на https://arxiv.org/pdf/1910.10261.pdf), в качестве акустических призанков MFCC. В проуессе обучения использовали следующие аугоментации: временной сдвиг,  "SpecAugment", SpecCutout. \
(3) https://arxiv.org/pdf/2003.12266v3.pdf - предложили некоторые способы различные виды attention'a (по акустическим признакам, по времени, и в композиции).

## Краткое описание выбранной архитектуры и выбранной метрики качества
В качестве архитектуры выбрад CNN-LSTM с Frequential attention (3). Однако отличия от статьи, в том что я произвожу attention с помощью сверток, а не размножений выходов пулингов в разрезе времени.

В процессе обучения использую смешивание сигнала с шумом (датасет шумов MUSAN).
В качестве ключевой метрики roc auc score.

Обучение производиться на данных train-clean-360 c добавлением шума из musan (free_noise + sound_bible), валидация на train-clean-100 + c добавлением шума из musan (free_noise + sound_bible), которых не было в обучающей выборке.\

Обучение проводил на cpu (1 эпоха 3 часа 15 минут).
## Prediction 
Файл predictions.json - {'file_name1':[score1,score2,...,scoreN_1], 'file_name2':[score1,score2,...,scoreN_2],...}


