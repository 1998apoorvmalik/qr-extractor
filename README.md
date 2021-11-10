# QR Extractor
Deep Learning based Image Segmentation Model to extract QR code regions from an Image.

Python QR code scanning libraries like Pyzbar sometimes don't work with raw QR code image because the QR code in the image could be very small, hence it becomes necessary to first identify and crop the QR region before processing it. This package allows to extract QR code regions from a given image so that QR scanners can more accurately identify and decode the QR codes.
