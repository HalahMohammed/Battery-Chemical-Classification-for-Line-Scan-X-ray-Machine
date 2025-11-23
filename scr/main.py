import halcon as ha


MaxImagesRegions=2
ImageWidth=1536
ImageHeight=547
TiledImage = ha.gen_image_const('uint2', ImageWidth, ImageHeight *MaxImagesRegions )
PrevRegions = ha.gen_empty_region()
ClipsProcessedSoFar= 0



def get_preprocess_params(model):
    proc = ha.HDevProcedure.load_external('create_dl_preprocess_param_from_model')
    call = ha.HDevProcedureCall(proc)
    call.set_input_control_param_by_name('DLModelHandle', model)
    call.set_input_control_param_by_name('NormalizationType', 'none')
    call.set_input_control_param_by_name('DomainHandling', 'full_domain')
    call.set_input_control_param_by_name('SetBackgroundID', [])
    call.set_input_control_param_by_name('ClassIDsBackground', [])
    call.set_input_control_param_by_name('GenParam', [])
    call.execute()
    return call.get_output_control_param_by_name('DLPreprocessParam')


def generate_samples(image):
    proc = ha.HDevProcedure.load_external('gen_dl_samples_from_images')
    call = ha.HDevProcedureCall(proc)
    call.set_input_iconic_param_by_name('Images', image)
    call.execute()
    return call.get_output_control_param_by_name('DLSampleBatch')


def preprocess_samples(samples, preprocess_param):
    proc = ha.HDevProcedure.load_external('preprocess_dl_samples')
    call = ha.HDevProcedureCall(proc)
    call.set_input_control_param_by_name('DLSampleBatch', samples)
    call.set_input_control_param_by_name('DLPreprocessParam', preprocess_param)
    call.execute()
    return samples  


def run_model_inference(model, samples):
    result_batch = ha.apply_dl_model(model, samples, [])
    class_names = ha.get_dict_tuple(result_batch, 'classification_class_names')
    class_name = ha.tuple_select(class_names, 0)
    print("Predicted class is:", class_name)
    if class_name in ["LITHIUM_D"",NIKEL_D","ALKALINE_D"]:

        depth=3
    elif class_name in ["LITHUM AAA","NIKELAAA","ALKALINE_AAA"]:
        depth=1
    else:
        depth=2


    return class_name


def main():

    
    image = ha.read_image("C:/Users/halah/Downloads/codes/Battery chemical classification/image024.tif")
    CurrRegions = ha.threshold(image, 0, 15000)
    clip_candidates = ha.connection(CurrRegions)
    FinishedClips = ha.select_shape (clip_candidates,  ['area','width'], 'and', [1000,20 ], [30000, 760])
    object_counted = ha.count_obj(FinishedClips)
    #print("number of detected shapes-->",FinishedClip
    ClipsInCurrentImageCoordinates = ha.move_region (FinishedClips, -ImageHeight, 0)
     #ha.dev_set_part (0, 0, (MaxImagesRegions + 1) * ImageHeight - 1, ImageWidth - 1)
    ClipsInTiledImageCoordinates = ha.move_region (FinishedClips, (MaxImagesRegions - 1) * ImageHeight, 0)
    print("Number of detected shapes:", object_counted)
    num_objects = ha.count_obj(FinishedClips)
    print(f"Number of found objects: {num_objects}")

    model = ha.read_dl_model('C:/Users/halah/Downloads/codes/Battery chemical classification/Classification Models/newclassifier.hdl')
    ha.set_dl_model_param(model, 'optimize_for_inference', 'true')
    ha.set_dl_model_param(model, 'batch_size', 1)

    preprocess_param = get_preprocess_params(model)

    for i in range(num_objects):
        object_selected = ha.select_obj( FinishedClips, i + 1)
        row, col, phi, l1, l2 = ha.smallest_rectangle2(object_selected)
        cropped_image=ha.crop_rectangle2( image,row, col, phi, l1, l2, 'true', 'constant')

        samples = generate_samples(cropped_image)
        samples = preprocess_samples(samples, preprocess_param)
        result = run_model_inference(model, samples)


if __name__ == "__main__":
    main()
