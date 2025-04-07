from pydub import AudioSegment
import os
import sys

def convert_m4a_to_wav(input_file, output_file=None):
    """
    Convert an .m4a file to .wav format
    
    Args:
        input_file (str): Path to the input .m4a file
        output_file (str, optional): Path for the output .wav file. If not specified,
                                    it will use the same name as the input file but with .wav extension
    
    Returns:
        str: Path to the output .wav file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return None
    
    # Check if input file is an m4a file
    if not input_file.lower().endswith('.m4a'):
        print(f"Error: Input file '{input_file}' is not an .m4a file")
        return None
    
    # If output file is not specified, create one based on input file
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.wav'
    
    try:
        # Load the m4a file
        audio = AudioSegment.from_file(input_file, format="m4a")
        
        # Export as wav
        audio.export(output_file, format="wav")
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        return output_file
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return None

if __name__ == "__main__":
    # If script is run directly, check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python convert.py <input_file.m4a> [output_file.wav]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_m4a_to_wav(input_file, output_file)