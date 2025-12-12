# Computer_Science_for_Business_Analytics
This repository contains the implementation of a duplicate detection method for web shop products, developed as part of the course Computer Science for Business Analytics at Erasmus University Rotterdam. The goal of this project is to design a scalable method that detects duplicate products based solely on product titles, using a combination of model words, character shingles, MinHashing, and Locality-Sensitive Hashing (LSH).

Although the method achieves strong scalability — reducing comparisons to ~2.45% of all product pairs — the accuracy remains low. This repository documents the full pipeline, enabling further experimentation or extension.

The structure of the code is as follows:
- Data cleaning;
- Feature extraction (model words and shingles);
- MinHashing
- LSH;
- Duplicate Classification.
