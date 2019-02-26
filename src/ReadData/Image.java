package ReadData;

import java.io.IOException;
import java.io.InputStream;

public class Image extends TestCase {
	/* number of rows and columns */
	int R, C;
	/* intensities from 0 to 255 in grayscale */
	int [][] gray;
	String name;
	/* label: >= 0 indicates a valid label for our classification NNet */
	int label;
	Image(int R, int C, InputStream in) throws IOException
	{
		
		label = -1;
		this.R = R;
		this.C = C;
		gray = new int[R][C];
		for (int i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
			{
				gray[i][j] = in.read();
				assert (gray[i][j]>=0);
			}
	}
	Image(int R, int C, InputStream inInputs, InputStream inLabels) throws IOException
	{
		this(R,C,inInputs);
		label = 1+inLabels.read();
		assert(label<=10);
	}
	/* reduce the size of an image */
	/* helps accelerate the speed of training */
	public void reduceImage(int R, int C)
	{
		assert R <= this.R;
		assert C <= this.C;

		int [][] gray = new int[R][C];
		int [][] cnt = new int[R][C];
		for (int i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
			{
				gray[i][j] = 0;
				cnt[i][j] = 0;
			}
		for (int i = 0; i < this.R; i++)
			for (int j = 0; j < this.C; j++)
			{
				int i2 = (i*R)/this.R, j2 = (j*C)/this.C;
				gray[i2][j2] += this.gray[i][j];
				cnt[i2][j2] += 1;
			}
		
		for (int i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
				if (cnt[i][j]>0)
					gray[i][j] /= cnt[i][j];
		
		this.gray = gray;
		this.C = C;
		this.R = R;
	}
	/* turns an image from grayscale to all 0 or 255 */
	public void blackWhite()
	{
		for (int i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
				if (gray[i][j]>125)
					gray[i][j] = 255;
				else
					gray[i][j] = 0;
	}

	@Override
	public double [] getTestData()
	{
		double [] ans = new double[R*C+1];
		ans[0] = 1.0; // bias unit of the input
		for (int i = 0; i < R; i++)
			for (int j = 0; j < C; j++)
				/* normalize to the interval [0,1] */
				ans[i*C + j+1] = gray[i][j] / 255.0;
		return ans;
	}
	@Override
	public int getTestLabel()
	{
		return label;
	}
}
