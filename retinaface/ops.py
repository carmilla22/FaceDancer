from retinaface.anchor import decode_tf, prior_box_tf
import tensorflow as tf

'''
ops.py è il modulo che prende le uscite grezze del detector RetinaFace e le trasforma in rilevamenti finali pronti per l’uso. 
In pratica riceve le predizioni di bounding box, landmark e probabilità dal modello, costruisce le prior box 
in base alla dimensione dell’immagine, decodifica le regressioni rispetto a queste ancore e applica una non-max suppression 
per eliminare i riquadri ridondanti. Questo significa che ops.py fa da ponte tra il formato interno del modello e il risultato 
finale: non crea né addestra il detector, ma prende le sue uscite e le converte in coordinate di box e landmark pulite, 
filtrate in modo che restino solo i volti più affidabili.
'''


def extract_detections(bbox_regressions, landm_regressions, classifications, image_sizes, iou_th=0.4, score_th=0.02):
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    variances = [0.1, 0.2]
    preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
        [bbox_regressions,
         landm_regressions,
         tf.ones_like(classifications[:, 0][..., tf.newaxis]),
         classifications[:, 1][..., tf.newaxis]], 1)
    priors = prior_box_tf(image_sizes, min_sizes, steps, False)
    decode_preds = decode_tf(preds, priors, variances)

    selected_indices = tf.image.non_max_suppression(
        boxes=decode_preds[:, :4],
        scores=decode_preds[:, -1],
        max_output_size=tf.shape(decode_preds)[0],
        iou_threshold=iou_th,
        score_threshold=score_th)

    out = tf.gather(decode_preds, selected_indices)

    return out

