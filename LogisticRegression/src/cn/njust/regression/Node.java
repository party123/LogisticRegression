package cn.njust.regression;


/**
 * 为了方便表示theta
 * 
 * 
 */
class Node {
	private double x0, x1, x2;

	public Node(double x0, double x1, double x2) {
		this.setX0(x0);
		this.setX1(x1);
		this.setX2(x2);
	}

	public Node() {
	}

	public double getX0() {
		return x0;
	}

	public void setX0(double x0) {
		this.x0 = x0;
	}

	public double getX1() {
		return x1;
	}

	public void setX1(double x1) {
		this.x1 = x1;
	}

	public double getX2() {
		return x2;
	}

	public void setX2(double x2) {
		this.x2 = x2;
	}
}