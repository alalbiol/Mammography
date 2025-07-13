import caffe
import numpy as np
import os
import argparse
import sys

def extract_caffe_weights(prototxt_path, model_path, output_dir, save_combined=True):
    """
    Loads a Caffe model and saves its weights and biases to .npy files.
    Optionally saves all parameters into a single .npz file.
    """
    if not os.path.exists(prototxt_path):
        print("Error: Prototxt file not found at {}".format(prototxt_path))
        return
    if not os.path.exists(model_path):
        print("Error: Caffe model file not found at {}".format(model_path))
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Output directory: {}".format(output_dir))

    # Set Caffe to CPU mode for weight extraction
    caffe.set_mode_cpu()
    print("Caffe set to CPU mode.")

    try:
        print("Loading Caffe model from {} with prototxt {}...".format(model_path, prototxt_path))
        net = caffe.Net(prototxt_path, model_path, caffe.TEST)
        print("Caffe model loaded successfully.")
    except Exception as e:
        print("Error loading Caffe model: {}".format(e))
        return

    print("\nExtracting and saving weights and biases...")
    combined_dict = {}

    for layer_name, param_list in net.params.items():
        if len(param_list) > 0:
            weights = param_list[0].data
            weights_filename = os.path.join(output_dir, "{}_weights.npy".format(layer_name))
            file_directory = os.path.dirname(weights_filename)
            if not os.path.exists(file_directory):
                os.makedirs(file_directory, exist_ok=True)
            np.save(weights_filename, weights)
            print("  Saved {} (shape: {})".format(os.path.basename(weights_filename), weights.shape))
            if save_combined:
                combined_dict["{}_weights".format(layer_name)] = weights

            if len(param_list) > 1:
                biases = param_list[1].data
                biases_filename = os.path.join(output_dir, "{}_biases.npy".format(layer_name))
                file_directory = os.path.dirname(biases_filename)
                if not os.path.exists(file_directory):
                    os.makedirs(file_directory, exist_ok=True)
                np.save(biases_filename, biases)
                print("  Saved {} (shape: {})".format(os.path.basename(biases_filename), biases.shape))
                if save_combined:
                    combined_dict["{}_biases".format(layer_name)] = biases
        else:
            print("  Layer {} has no learnable parameters.".format(layer_name))

    if save_combined:
        combined_file = os.path.join(output_dir, "all_weights_biases.npz")
        np.savez(combined_file, **combined_dict)
        print("\nCombined parameters saved to: {}".format(combined_file))

    print("\nAll weights and biases extracted to: '{}'".format(output_dir))

def main():
    parser = argparse.ArgumentParser(description="Extract weights and biases from a Caffe model.")
    parser.add_argument("--prototxt", type=str, help="Path to the Caffe prototxt file.")
    parser.add_argument("--model", type=str, help="Path to the Caffe .caffemodel file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save extracted weights.")
    parser.add_argument("--no_combined", action="store_true", help="Disable saving a single combined .npz file.")

    args = parser.parse_args()

    if not args.prototxt or not args.model:
        print("Error: --prototxt and --model arguments are required.")
        parser.print_help()
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else "extracted_caffe_weights"
    save_combined = not args.no_combined

    extract_caffe_weights(args.prototxt, args.model, output_dir, save_combined)

if __name__ == "__main__":
    main()
