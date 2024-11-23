#!/bin/bash
# https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
wget -O loan-approval.zip https://www.kaggle.com/api/v1/datasets/download/taweilo/loan-approval-classification-data
unzip -o loan-approval.zip
rm loan-approval.zip