package ac.up.cos711.rbfnntraining.data;

import java.util.Arrays;

/**
 * Class that represents a single line of the dataset (a data pattern). All
 * methods of this class use deep copying for proper composition.
 *
 * @author Abrie van Aardt
 */
public class Pattern {

    public double[] getInputs() {
        return Arrays.copyOf(inputs, inputs.length);
    }

    public double[] getTargets() {
        return Arrays.copyOf(targets, targets.length);
    }

    public void setInputs(double[] _inputs) {
        inputs = Arrays.copyOf(_inputs, _inputs.length);
    }

    public void setTargets(double[] _targets) {
        targets = Arrays.copyOf(_targets, _targets.length);
    }

    private double[] inputs;
    private double[] targets;
}
