## Research notes
Discussion [Link](https://github.com/yinjunbo/ProficientTeachers/issues)

### ProfficientTeacher
```
We used 4 GPUs with total batch size of 16 and learning rate 0.003. We trained 30 epochs.
Sometimes training results may be unstable for ped. and cyc., but stable for car, which is around 49 mAPH.
Here's the training list for 5% setting that you can double check with:
'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord',
'segment-10750135302241325253_180_000_200_000_with_camera_labels.tfrecord',
'segment-11718898130355901268_2300_000_2320_000_with_camera_labels.tfrecord',
'segment-1265122081809781363_2879_530_2899_530_with_camera_labels.tfrecord',
'segment-13519445614718437933_4060_000_4080_000_with_camera_labels.tfrecord',
'segment-14348136031422182645_3360_000_3380_000_with_camera_labels.tfrecord',
'segment-15266427834976906738_1620_000_1640_000_with_camera_labels.tfrecord',
'segment-1605912288178321742_451_000_471_000_with_camera_labels.tfrecord',
'segment-16735938448970076374_1126_430_1146_430_with_camera_labels.tfrecord',
'segment-17752423643206316420_920_850_940_850_with_camera_labels.tfrecord',
'segment-1891390218766838725_4980_000_5000_000_with_camera_labels.tfrecord',
'segment-2570264768774616538_860_000_880_000_with_camera_labels.tfrecord',
'segment-3195159706851203049_2763_790_2783_790_with_camera_labels.tfrecord',
'segment-3919438171935923501_280_000_300_000_with_camera_labels.tfrecord',
'segment-4672649953433758614_2700_000_2720_000_with_camera_labels.tfrecord',
'segment-5458962501360340931_3140_000_3160_000_with_camera_labels.tfrecord',
'segment-6242822583398487496_73_000_93_000_with_camera_labels.tfrecord',
'segment-7187601925763611197_4384_300_4404_300_with_camera_labels.tfrecord',
'segment-8031709558315183746_491_220_511_220_with_camera_labels.tfrecord',
'segment-9016865488168499365_4780_000_4800_000_with_camera_labels.tfrecord'
```