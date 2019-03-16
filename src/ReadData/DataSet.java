package ReadData;

import java.util.Random;

/* TODO: use ArrayList instead of an actual array? 
 * this would simplify the addition of the bias unit
 * outside of this context */

abstract public class DataSet {
	/* number of test cases */
	int M;
	int bdx = 0;
	private Random random = new Random();
	/* test cases */
	TestCase [] cases;
	public TestCase [] getTrainingCases()
	{
		return cases;
	}
	public void shuffle()
	{
		for (int k = M-1; k > 0; k--)
		{
			int j = random.nextInt(k);
			TestCase aux = cases[j];
			cases[j] = cases[k-1];
			cases[k-1] = aux;
		}
		
	}
	public TestCase [] getBatch(int batchSize)
	{
		TestCase [] a = new TestCase[batchSize];
		for (int j = 0; j < batchSize; j++)
			a[j] = cases[(bdx+j)%M];
		bdx = (bdx+batchSize)%M;
		return a;
	}

}
