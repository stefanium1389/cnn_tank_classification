## CNN Tank Classification

# Аутор: Стефан Богдановић, SV44/2020

# Проблем
Проблем је препознавање тенкова из Другог светског рата. Решаван употребом конволутивне неуронске мреже

# Покретање
Потребно је само покренути main.py након прављења виртуелног окружења и инсталирања свих библиотека
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

библиотеке које се користе 



