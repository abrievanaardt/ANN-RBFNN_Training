package ac.up.cos711.rbfnntraining.data.util;

/**
 *
 * @author Abrie van Aardt
 */
public class IncorrectFileFormatException extends Exception {
    @Override
    public String getMessage(){
        return "Make sure the format of the dataset file is correct.";
    }
}
