from django.db import models


class Review(models.Model):
    title = models.CharField(max_length=200)
    review = models.CharField(max_length=200)
    score = models.CharField(max_length=3)