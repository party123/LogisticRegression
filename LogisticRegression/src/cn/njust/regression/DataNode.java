package cn.njust.regression;

/**
 * 记录数据集里面的每一个数据及其分类，增加x0=1.0
 * 
 * 
 */
class DataNode {
	private double x0, x1, x2, y;

	public DataNode(double x0, double x1, double x2) {
		this.x0 = x0;
		this.x1 = x1;
		this.x2 = x2;
	}

	public DataNode() {
	}

	public void Print() {
		System.out.println("当前的x1值为：" + this.x1 + ";当前的x2值为：" + this.x2);
	}

	public double getX1() {
		return this.x1;
	}

	public double getX2() {
		return this.x2;
	}

	public void setX1(double x1) {
		this.x1 = x1;
	}

	public void setX2(double x2) {
		this.x2 = x2;
	}

	public double getX0() {
		return x0;
	}

	public void setX0(double x0) {
		this.x0 = x0;
	}

	public double getY() {
		return y;
	}

	public void setY(double y) {
		this.y = y;
	}
}