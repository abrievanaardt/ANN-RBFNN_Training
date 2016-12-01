package ac.up.cos711.rbfnntraining.function.util;

/**
 *
 * @author Abrie van Aardt
 */
public class NotAFunctionException extends Exception{
    @Override
    public String getMessage(){
        return "This class does not implement the IFunction interface";
    }
}
