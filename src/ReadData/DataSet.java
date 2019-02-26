package ReadData;

/* TODO: use ArrayList instead of an actual array? 
 * this would simplify the addition of the bias unit
 * outside of this context */

abstract public class DataSet {
	/* number of test cases */
	int M;
	/* test cases */
	TestCase [] cases;
	public TestCase [] getTrainingCases()
	{
		return cases;
	}
}
