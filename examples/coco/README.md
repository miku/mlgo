# Pretrained model

* Download: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

```
$ sha1sum ssd_mobilenet_v1_coco_11_06_2017.tar.gz
a88a18cca9fe4f9e496d73b8548bfd157ad286e2  ssd_mobilenet_v1_coco_11_06_2017.tar.gz
```

If using the GPU is not possible due to any version mismatch, try using just the CPU:

```
$ CUDA_VISIBLE_DEVICES=-1
```

----

Example images:

* Umbrellas,
  [Source](https://firenze.repubblica.it/cronaca/2013/12/04/foto/gli_ombrelli_sospesi_di_via_romana-72688861/1/#1),
© Divisione [La Repubblica](https://repubblica.it) Gruppo Editoriale L'Espresso Spa - P.Iva 00906801006.
* Florence bus (Photo by Hubert Gajewski), via [Florence buses](http://www.reidsitaly.com/destinations/tuscany/florence/planning/around-by-bus.html)
