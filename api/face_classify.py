import datetime
from pathlib import Path

import numpy as np
import pytz
import seaborn as sns
from django.core.cache import cache
from django.db import connection
from django_rq import job
from sklearn import svm
from sklearn.decomposition import PCA

from api.distance import findEuclideanDistance
from api.models import Face, Person, LongRunningJob, Photo
from api.util import logger


def cluster_faces(user):
    # for front end cluster visualization

    people = [
        p.id
        for p in Person.objects.filter(faces__photo__owner=user).distinct()
    ]
    colors = sns.color_palette('Dark2', len(people)).as_hex()
    p2c = dict(zip(people, colors))

    faces = Face.objects.filter(photo__owner=user)
    face_encodings_all = []
    for face in faces:
        face_encoding = np.frombuffer(bytes.fromhex(face.encoding))
        face_encodings_all.append(face_encoding)

    pca = PCA(n_components=3)
    vis_all = pca.fit_transform(np.array(face_encodings_all))
    res = []
    for face, vis in zip(faces, vis_all):
        person_id = face.person.id  # color
        person_name = face.person.name
        person_label_is_inferred = face.person_label_is_inferred
        face_url = face.image.url
        value = {'x': vis[0], 'y': vis[1], 'size': vis[2]}
        out = {
            "person_id": person_id,
            "person_name": person_name,
            "person_label_is_inferred": person_label_is_inferred,
            "color": p2c[person_id],
            "face_url": face_url,
            "value": value
        }
        res.append(out)
    return res


def findEuclideanDistance(source_representation, test_representation):
    source_representation = l2_normalize(source_representation)
    test_representation = l2_normalize(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


@job
def train_faces(user, job_id):
    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_TRAIN_FACES)
        lrj.save()

    try:
        persons = Person.objects.exclude(name="unknown")
        persons_list = []
        person_face_encodings = []
        for person in persons:
            face_encoding = np.frombuffer(bytes.fromhex(person.mean_face_encoding))
            person_face_encodings.append(face_encoding)
            persons_list.append(person)
        faces = Face.objects.filter(
            photo__owner=user).prefetch_related('person')
        if len(persons) == 0:
            logger.info("No labeled faces found")
            lrj.finished = True
            lrj.failed = False
            lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
            lrj.save()
            return True

        target_count = len(faces)
        for idx, face in enumerate(faces):
            if face.person_label_is_inferred in [True, None] or face.person.name == 'unknown':
                face_encoding = np.frombuffer(bytes.fromhex(face.encoding))
                # results = face_recognition.face_distance(person_face_encodings, face_encoding)
                results = [findEuclideanDistance(x, face_encoding) for x in person_face_encodings]
                TOLERANCE = 0.35
                person_idx = np.argmin(results)
                logger.info(f"{face.image_path}, {idx}/{len(faces)}, cosine: {results[person_idx]}")
                if results[person_idx] <= TOLERANCE:
                    person = Person.objects.get(name=persons_list[person_idx].name)
                    logger.info(person.name)
                    face.person = person
                    face.person_label_is_inferred = True
                    face.person_label_probability = float(results[person_idx])
                    face.save()
                else:
                    person = Person.objects.get(name="unknown")
                    face.person = person
                    face.person_label_is_inferred = None
                    face.person_label_probability = 0
                    face.save()

            lrj.result = {
                'progress': {
                    "current": idx + 1,
                    "target": target_count
                }
            }
            lrj.save()

        lrj.finished = True
        lrj.failed = False
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
        cache.clear()
        return True

    except BaseException:
        logger.exception("An error occured")
        res = []

        lrj.failed = True
        lrj.finished = True
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
        return False

    return res


@job
def train_faces_svm(user, job_id):
    if LongRunningJob.objects.filter(job_id=job_id).exists():
        lrj = LongRunningJob.objects.get(job_id=job_id)
        lrj.started_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
    else:
        lrj = LongRunningJob.objects.create(
            started_by=user,
            job_id=job_id,
            queued_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            started_at=datetime.datetime.now().replace(tzinfo=pytz.utc),
            job_type=LongRunningJob.JOB_TRAIN_FACES)
        lrj.save()

    try:
        encodings = []
        persons = []
        faces = Face.objects.filter(photo__owner=user).prefetch_related('person')
        # get the faces
        for idx, face in enumerate(faces):
            if face.person_label_is_inferred is False and face.person.name != 'unknown':
                face_encoding = np.frombuffer(bytes.fromhex(face.encoding))
                encodings.append(face_encoding)
                persons.append(face.person.name)
        logger.info(persons)
        if len(persons) == 0:
            logger.info("No labeled faces found")
            lrj.finished = True
            lrj.failed = False
            lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
            lrj.save()
            return True

        logger.info("running svm classification")
        clf = svm.SVC(gamma="scale")
        clf.fit(encodings, persons)

        target_count = len(faces)
        for idx, face in enumerate(faces):
            if face.person_label_is_inferred in [True, None] or face.person.name == 'unknown':
                face_encoding = np.frombuffer(bytes.fromhex(face.encoding))
                name = clf.predict([face_encoding])
                person = Person.objects.get(name=name[0])
                face.person = person
                face.person_label_is_inferred = True
                face.save()

            lrj.result = {
                'progress': {
                    "current": idx + 1,
                    "target": target_count
                }
            }
            lrj.save()

        lrj.finished = True
        lrj.failed = False
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
        cache.clear()
        return True

    except BaseException:
        logger.exception("An error occured")
        res = []

        lrj.failed = True
        lrj.finished = True
        lrj.finished_at = datetime.datetime.now().replace(tzinfo=pytz.utc)
        lrj.save()
        return False

    return res


def reset_faces():
    logger.info('deleteing faces')
    cursor = connection.cursor()
    cursor.execute("TRUNCATE api_face")
    images = list(Path('/protected_media/faces').glob('**/*.jpg'))
    for image in images:
        image.unlink()
    logger.info(f"removed {len(images)} images")
    cache.clear()

    logger.info('processing photos')
    for i, photo in enumerate(Photo.objects.all()):
        photo._extract_faces()
        logger.info(f'finish extracting faces, {i}')
        photo.save()


if __name__ == "__main__":
    res = train_faces()
