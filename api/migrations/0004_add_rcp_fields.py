from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('api', '0003_remove_unused_thumbs'),
    ]

    operations = [
        migrations.AddField(
            model_name='Face',
            name='emotion',
            field=models.TextField()
        ),
        migrations.AddField(
            model_name='Photo',
            name='bounding_box_image',
            field=models.ImageField(upload_to='bounding_box_images')
        ),
        migrations.AddField(
            model_name='Photo',
            name='text_encoding',
            field=models.TextField(default=None, null=True)
        )
    ]
