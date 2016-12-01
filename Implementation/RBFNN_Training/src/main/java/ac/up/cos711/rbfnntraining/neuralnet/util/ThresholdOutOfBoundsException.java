package ac.up.cos711.rbfnntraining.neuralnet.util;

/**
 *
 * @author Abrie van Aardt
 */
public class ThresholdOutOfBoundsException extends Exception{
    @Override
    public String getMessage(){
        return "The threshold specified is not within acceptable bounds.";
    }
}
