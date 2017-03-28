#include <iostream>
#include <iterator>
#include <string>
#include <stack>
#include <queue>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <tuple>
#include <functional>
#include <assert.h>


using namespace std;
//enjoy your talents

template<typename T>
void insertionSort(vector<T>&arr)//CLRS-P10
{
	for (int i = 1; i <= static_cast<int>(arr.size())-1; ++i)//size()���ص���size_t������Ҫ�Ȱ���תΪint
	{
		T curVal = arr[i];
		int j = i - 1;
		for (; j >= 0; --j)//���ﲻ����size_t j����Ϊ��j=0ʱ��--j��Ľ��������-1������
		{
			if (curVal <= arr[j])arr[j + 1] = arr[j];
			else break;//����for...break�ṹ�����Ը�Ϊwhile����ʽ
		}
		arr[j + 1] = curVal;
	}
}

template<typename T>
void merge(vector<T>&arr, int p, int q, int r)//CLRS-P17
{
	//p<=q<q+1<=r
	vector<T>la(arr.begin()+p,arr.begin()+q+1);//[p,q]
	vector<T>ra(arr.begin()+q+1,arr.begin()+r+1);//[q+1,r]
	la.push_back(INT_MAX);
	ra.push_back(INT_MAX);
	int i = 0, j = 0;
	for (int k = p; k <= r; ++k)
	{
		arr[k] = (la[i] < ra[j] ? la[i++] : ra[j++]);
	}
}
template<typename T>
void mergeSort2(vector<T>&arr,int p,int r)//CLRS-P19
{
	if (r-p+1 <= 1)return;
	int q = (r + p) / 2;//ע�ⲻ��q=(r-p)/2
	mergeSort2(arr, p, q);
	mergeSort2(arr,q+1,r);
	merge(arr,p,q,r);
}
template<typename T>
void mergeSort(vector<T>&arr)
{
	mergeSort2(arr,0,arr.size()-1);
}

template<typename T>
tuple<int, int, T> findMaxCrossingSubarray(const vector<T>&arr,int left,int right,int mid)//CLRS-P40
{
	//left<=mid<mid+1<=right
	int maxLeft, maxRight;
	T maxLeftSum = INT_MIN, maxRightSum = INT_MIN;
	T curLeftSum=0, curRightSum=0;
	for (int i = mid; i >= left; --i)//[left,mid]
	{
		curLeftSum += arr[i];
		if (curLeftSum > maxLeftSum)
		{
			maxLeftSum = curLeftSum;
			maxLeft = i;
		}
	}
	for (int i = mid + 1; i <= right; ++i)//[mid+1,right]
	{
		curRightSum += arr[i];
		if (curRightSum > maxRightSum)
		{
			maxRightSum = curRightSum;
			maxRight = i;
		}
	}
	return make_tuple(maxLeft, maxRight, maxLeftSum + maxRightSum);//��Ϊ�ǿ�Խ��crossing����mid����ʱmaxRight>mid(û�е���)
}

template<typename T>
tuple<int, int, T> findMaxSubarray(const vector<T>&arr, int left, int right)//CLRS-P41
{
	if (left == right)return make_tuple(left, right, arr[left]);
	int mid = (left + right) / 2;
	auto l = findMaxSubarray(arr,left,mid);//auto l = findMaxSubarray(arr,left,mid-1)�Ǵ���ģ�������һ��������˵�right=left+1ʱ,mid=(right+left)/2=left,��ômid-1=left-1<left�ˣ�����
	auto r = findMaxSubarray(arr,mid+1,right);
	auto m = findMaxCrossingSubarray(arr,left,right,mid);
	T lsum = get<2>(l), rsum = get<2>(r), msum = get<2>(m);
	if (lsum >= max(rsum, msum))return l;
	else if (rsum >= max(lsum, msum))return r;
	else return m;
}


//ֻ�����ȫ����������ȷ����������±꣬���Ҹ������0��ʼ��ŵ�
inline int PARENT(int i){ return (i + 1) / 2 - 1; }
inline int LEFT(int i){return  (i + 1) * 2 - 1; }
inline int RIGHT(int i){ return (i + 1) * 2; }
template<typename T>
void maxHeapifyRecursive(vector<T>&arr, int i, int indexOfLastNode)//CLRS-P86 ά����������,�ú�����һ��ǰ����ǽ��i��������������������
{
	//indexOfLastNode��ʾ�����һ��Ԫ�ص��±�
	//ȷ��i�������Һ����ĸ��Ƚϴ󣬰����ĺͽ��i����λ�ã�Ȼ���½�һ�㣬ֱ������һ��
	int l = LEFT(i), r = RIGHT(i);
	int indexOfLargestNode = i;
	if (l <= indexOfLastNode&&arr[l] > arr[indexOfLargestNode])indexOfLargestNode = l;
	if (r <= indexOfLastNode&&arr[r] > arr[indexOfLargestNode])indexOfLargestNode = r;
	if (indexOfLargestNode != i)
	{
		swap(arr[i],arr[indexOfLargestNode]);
		maxHeapifyRecursive(arr, indexOfLargestNode, indexOfLastNode);
	}
}
template<typename T>
void maxHeapify(vector<T>&arr, int i, int indexOfLastNode)
{
	int l, r, indexOfLargestNode;
	int indexOfFirstLeafNode = PARENT(indexOfLastNode) + 1;//��һ��Ҷ�ӽ���±�
	while (i < indexOfFirstLeafNode)//iΪ��Ҷ�ӽڵ�ʱ����ѭ��
	{
		l = LEFT(i), r = RIGHT(i);
		indexOfLargestNode = i;
		if (l <= indexOfLastNode&&arr[l] > arr[indexOfLargestNode])indexOfLargestNode = l;
		if (r <= indexOfLastNode&&arr[r] > arr[indexOfLargestNode])indexOfLargestNode = r;
		if (indexOfLargestNode == i)break;
		swap(arr[i], arr[indexOfLargestNode]);
		i = indexOfLargestNode;
	}
}
template<typename T>
void buildMaxHeap(vector<T>&arr)//CLRS-P87�����ѣ�����ʱ�临�Ӷ�
{
	//�ӵ�һ����Ҷ�ӽڵ㿪ʼ�����ҵ��󣬴��µ��ϣ�����ά�������ʵĺ���maxHeapify
	int indexOfLastNode = arr.size() - 1;
	int indexOfFirstNonleafNode = PARENT(indexOfLastNode);
	for (int i = indexOfFirstNonleafNode; i >= 0; --i)
		maxHeapify(arr,i,indexOfLastNode);
}
template<typename T>
void heapSort(vector<T>&arr)
{
	if (arr.empty())return;
	buildMaxHeap(arr);
	for (int i = arr.size() - 1; i >= 1; --i)
	{
		swap(arr[0], arr[i]);
		maxHeapify(arr, 0, i - 1);
	}
}
template<typename T>
int partition(vector<T>&arr, int left, int right)
{
	//�Ե�һ��Ԫ��Ϊ��Ԫ
	int l = left, r = right;
	T val = arr[left];
	while (l < r)
	{
		while (l < r&&val <= arr[r])--r;
		if (l == r)return l;
		swap(arr[l++], arr[r]);
		while (l < r&&val >= arr[l])++l;
		if (l == r)return r;
		swap(arr[r--],arr[l]);
	}
	return l;
}

template<typename T>
int partition2(vector<T>&arr, int left, int right)
{
	int i = left - 1;// rightmost index of left part
	T val = arr[right];
	for (int j = left; j <= right - 1; ++j)
		if (arr[j] <= val) swap(arr[++i], arr[j]);
	swap(arr[++i],arr[right]);
	return i;
}

template<typename T>
void quickSort(vector<T>&arr,int left,int right)
{
	if (right-left+1<=1)return;
	int p = partition2(arr,left,right);
	quickSort(arr,left,p-1);
	quickSort(arr,p+1,right);
}
//merge sort for linked list
struct listNode
{
	int val;
	listNode* next;
};
listNode* genList(vector<int>&arr)
{
	if (arr.empty())return NULL;
	listNode* cur, *pre, *head;
	head = (listNode*)malloc(sizeof(listNode));
	head->val = arr[0];
	head->next = NULL;
	pre = head;
	for (int i = 1; i < (int)arr.size();++i)
	{
		cur = (listNode*)malloc(sizeof(listNode));
		cur->val = arr[i];
		cur->next = NULL;
		pre->next = cur;
		pre = cur;
	}
	return head;
}
void printList(listNode*lstHead)
{
	while (lstHead!=NULL)
	{
		cout << lstHead->val << " ";
		lstHead = lstHead->next;
	}
	cout << endl;
}
listNode* getMid(listNode*head)
{
	listNode* fast = head, *slow = head, *preSlow = slow;
	while (fast != NULL&&fast->next != NULL)
	{
		fast = fast->next->next;
		preSlow = slow;
		slow = slow->next;
	}
	return preSlow;
}
listNode* merge(listNode*head1, listNode*head2)
{
	listNode*cl = head1, *cr = head2,*resHead,*pre;
	if ((cr == NULL) || (cr != NULL&&cl->val < cr->val))
	{
		resHead = cl;
		cl = cl->next;
	}
	else
	{
		resHead = cr;
		cr = cr->next;
	}
	pre = resHead;
	while (cl != NULL||cr != NULL)
	{
		if (cl != NULL)
		{
			if (cr == NULL || (cr != NULL&&cl->val < cr->val))
			{
				pre->next = cl;
				cl = cl->next;
			}
			else
			{
				pre->next = cr;
				cr = cr->next;
			}
		}
		else
		{
			pre->next = cr;
			cr = cr->next;
		}
		pre = pre->next;
	}
	return resHead;
}
listNode* mergeSort(listNode*head)
{
	if (head==NULL||head->next==NULL)return head;
	auto mid = getMid(head);
	listNode*head1 = head, *head2 = mid->next;
	mid->next = NULL;
	head1=mergeSort(head1);
	head2=mergeSort(head2);
	return merge(head1,head2);
}
vector<int> cntSort(const vector<int>&arr, int k)
{
	//arr�ķ�Χ��[0,k]��
	vector<int>cntArr(k+1,0);
	for (int a : arr)
		++cntArr[a];
	for (int i = 1; i < (int)cntArr.size(); ++i)
		cntArr[i] += cntArr[i - 1];
	vector<int>res(arr.size());
	for (int i = (int)(arr.size() - 1); i >= 0; --i)
	{
		res[cntArr[arr[i]] - 1] = arr[i];
		--cntArr[arr[i]];
	}
	return res;
}
void radixSort(vector<string>&arr, int d)	//���߷ֱʼǡ�P240
{
	//d��ʾ�ؼ���λ��,����100��3��λ����d=3
	//ʱ�临�Ӷ�ΪO��d(n+k)��,����k��ʾÿ���ؼ�λ�ϵ�ȡֵ��Χ����Ӧ�����qs.size(),��������������˵,k=10(0-9);
	vector<queue<string>>qs(10);
	//�ܹ�����d�֡�����+�ռ���
	for (int i = d - 1; i >= 0; --i)
	{
		//���������λ��ʼ
		//��i�ַ��䣨���䵽fifo����ȥ��
		for (auto& a : arr)
		{
			qs[a[i] - 48].push(a);
		}
		arr.clear();
		//��i���ռ�����fifo�������ռ���arr����ȥ��
		for (auto&q : qs)
		{
			while (!q.empty())
			{
				arr.push_back(q.front());
				q.pop();
			}
		}

	}
}
template<typename T>//CLRS-P120ѡ���kС��˳��ͳ����
T orderSelection(vector<T>&arr, int left,int right,int kth)
{
	if (left == right)return arr[left];//BASE CASE
	int p = partition2(arr,left,right);
	int k = p - left + 1;
	if (k == kth)return arr[p];
	else if (k > kth)return orderSelection(arr,left,p-1,kth);
	else return orderSelection(arr, p + 1, right, kth - k);
}
template<typename T>
struct gNode
{
	T info;
	int no;//node�ı��
};
template<typename T>
struct graph
{
	vector<gNode<T>*>nodes;
	vector<vector<int>>edges;
};
//��С�������ǰ������н��������б�Ȩ��֮����С
template<typename T>
int prim(graph<T>*G, int v0)
{
	//�Ա��no=v0�Ķ��㿪ʼѰ����С������
	int maxNo = G->nodes.size() - 1;//ͼ����������
	vector<bool>vset(maxNo+1,false);//��ʾ��Ӧ��ŵĶ����Ƿ��Ѽ�����С������
	vector<int>lowCost(maxNo+1);//��ʾ��Ӧ��ŵĶ��㵽��ǰ��С��������������̱ߵ�Ȩ��
	int resWeightSum = 0;
	vset[v0] = true;//v0��Ϊ��һ�������С������
	for (int i = 0; i <= maxNo; ++i)
	{
		//��ʼ�����㵽������ʵ���Ǳ��v0�ĵ㣩����̱�Ȩ��
		lowCost[i] = G->edges[v0][i];
	}
	
	for (int i = 0; i <= maxNo - 1; ++i)
	{
		//ÿ��ѭ���ҵ���û�������Ķ�����lowcost����С���Ǹ��㣬���������vset����������С�������У���ע��ÿ����һ�����㵽����������Ҫ����lowcost��
		//�ܹ���Ҫ�����v0���������,��Ҫѭ��maxNo��

		int k;//��¼�������Ķ�����
		int minWeight = INT_MAX;
		for (int j = 0; j <= maxNo; ++j)
		{
			if (!vset[j] && lowCost[j] < minWeight)
			{
				k = j;
				minWeight = lowCost[j];
			}
		}
		vset[k] = true;
		resWeightSum += minWeight;
		//����lowcost
		for (int j = 0; j <= maxNo; ++j)
		{
			if (!vset[j] && G->edges[k][j] < lowCost[j])
				lowCost[j] = G->edges[k][j];
		}
		
	}
	
	return resWeightSum;//����������С�������ĸ���Ȩ�صĺ�
}
template<typename T>
void dijkstra(graph<T>*G, int v0, vector<int>&dist, vector<int>&path)
{
	//path[i]���ڼ�¼i��㵽Դ���v0���·����ǰ��
	//dist[i]���ڼ�¼i��㵽Դ�ڵ�v0�����·������
	int maxNo = G->nodes.size() - 1;
	vector<bool>vset(maxNo+1,false);
	dist = vector<int>(maxNo + 1);
	path = vector<int>(maxNo+1,-1);
	for (int i = 0; i <= maxNo; ++i)
	{
		dist[i] = G->edges[v0][i];
		if (G->edges[v0][i] < INT_MAX)
			path[i] = v0;
	}
	vset[v0] = true;
	path[v0] = -1;//���㷨������ֻ��path[v0]=-1�����Ķ���ǰ�����ı��
	for (int i = 0; i <= maxNo - 1; ++i)
	{
		int k;
		int mindist = INT_MAX;
		for (int j = 0; j <= maxNo; ++j)
		{
			if (!vset[j] && dist[j] < mindist)
			{
				mindist = dist[j];
				k = j;
			}
		}
		vset[k] = true;
		for (int j = 0; j <= maxNo; ++j)
		{
			if (!vset[j] && dist[k] + G->edges[k][j] < dist[j])
			{
				dist[j] = dist[k] + G->edges[k][j];
				path[j] = k;
			}
		}
	}
}
void printPath(const vector<int>& path, int destination)
{
	stack<int>stk;
	stk.push(destination);
	int pre = path[destination];
	while (pre != -1)
	{
		stk.push(pre);
		pre = path[pre];
	}
	while (!stk.empty())
	{
		cout << stk.top() << " ";
		stk.pop();
	}
	cout << endl;
}
int memoizedCutRodAux(const vector<int>&arr, int n, vector<int>&memo)
{
	if (memo[n] != -1)return memo[n];
	int q = INT_MIN;
	for (int i = 1; i <= n; ++i)
	{
		q = max(q,arr[i]+memoizedCutRodAux(arr,n-i,memo));
	}
	memo[n] = q;
	return q;
}
int memoizedCutRod(const vector<int>&arr, int n)
{
	vector<int>memo(n+1,-1);//memo�Ĺ�ģ��n�ҹ�,memo���������洢����������Ľ��
	memo[0] = 0;
	return memoizedCutRodAux(arr,n,memo);
}
int bottomUpCutRod(const vector<int>&arr, int n)
{
	vector<int>r(n+1);
	r[0] = 0;
	for (int i = 1; i <= n; ++i)
	{
		int q = INT_MIN;
		for (int j = 1; j <= i; ++j)
		{
			q = max(q,arr[j]+r[i-j]);
		}
		r[i] = q;
	}
	return r[n];
}
int matrixChainOrder(const vector<int>&arr, vector<vector<int>>&s)
{
	int n = arr.size() - 1;
	vector<vector<int>>dpMat(1+n,vector<int>(1+n));//dp[i][j]��ʾAi....Aj�����Ҫ��������
	s = dpMat;
	for (int i = 1; i <= n; ++i)dpMat[i][i] = 0;
	for (int l = 2; l <= n; ++l)
	{
		//�����l��������ѭ���������棬��ʾ�ȼ����������̵�������
		for (int i = 1; i <= n - l + 1; ++i)
		{
			int m = INT_MAX;
			int mi;
			int j = i + l - 1;
			for (int k = i; k <= j-1; ++k)
			{
				if (m > dpMat[i][k] + arr[i - 1] * arr[k] * arr[j] + dpMat[k + 1][j])
				{
					m = dpMat[i][k] + arr[i - 1] * arr[k] * arr[j] + dpMat[k + 1][j];
					mi = k;
				}
			}
			dpMat[i][j] = m;//�������������һ�������˴���������������
			s[i][j] = mi;
		}
	}
	return dpMat[1][n];
}
int lookupChain(const vector<int>&arr, vector<vector<int>>&s, vector<vector<int>>&memo, int i, int j)
{
	if (memo[i][j] != -1)return memo[i][j];
	int m = INT_MAX;
	int mi;
	int cm;
	for (int k = i; k <= j - 1; ++k)
	{
		cm = lookupChain(arr, s, memo, i, k) + lookupChain(arr, s, memo, k + 1, j) + arr[i - 1] * arr[k] * arr[j];
		if (m >cm )
		{
			m = cm;
			mi = k;
		}
	}
	memo[i][j] = m;
	s[i][j] = mi;
	return m;
}
int memoizedMatrixChain(const vector<int>&arr, vector<vector<int>>&s)
{
    int n = arr.size() - 1;
	vector<vector<int>>memo(1 + n, vector<int>(1 + n,-1));//dp[i][j]��ʾAi....Aj�����Ҫ��������
	s = vector<vector<int>>(1+n,vector<int>(1+n));
	for (int i = 1; i <= n; ++i)memo[i][i] = 0;
	return lookupChain(arr,s,memo,1,n);
}
void printOptimalParens(const vector<vector<int>>s, int i, int j)
{
	if (i == j)
		cout << "A" << i;
	else
	{
		cout << "(";
		printOptimalParens(s, i, s[i][j]);
		printOptimalParens(s,s[i][j]+1,j);
		cout << ")";
	}
}
vector<char> LCS_COSTLY(const vector<char>&X, const vector<char>&Y)
{
	int m = X.size();//X:<x0,x1....x(m-1)>
	int n = Y.size();
	typedef vector<char> seq;
	vector<vector<seq>>dpm(m+1,vector<seq>(n+1,seq()));
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (X[i - 1] == Y[j - 1])
			{
				dpm[i][j] = dpm[i - 1][j - 1];
				dpm[i][j].push_back(X[i-1]);
			}
			else
			{
				if (dpm[i - 1][j].size() > dpm[i][j - 1].size())
				{
					dpm[i][j] = dpm[i - 1][j];
				}
				else
				{
					dpm[i][j] = dpm[i][j-1];
				}
			}
		}
	}
	return dpm[m][n];
}
int LCS(const vector<char>&X, const vector<char>&Y, vector<vector<char>>&b)
{
	//����lcs�ĳ���
	int m = X.size();//X:<x0,x1....x(m-1)>
	int n = Y.size();
	vector<vector<int>>c(m+1,vector<int>(n+1,0));
	b = vector<vector<char>>(m+1,vector<char>(n+1));
	for (int i = 1; i <= m; ++i)
	{
		for (int j = 1; j <= n; ++j)
		{
			if (X[i - 1] == Y[j - 1]){ 
				c[i][j] = c[i - 1][j - 1] + 1;
				b[i][j] = 'd';
			}
			else { 
				if (c[i - 1][j] > c[i][j - 1])
				{
					c[i][j] = c[i - 1][j];
					b[i][j] = 'u';
				}
				else
				{
					c[i][j] = c[i][j-1];
					b[i][j] = 'l';
				}
			}
		}
	}
	return c[m][n];
}
void printLCS(const vector<char>& X, const vector<vector<char>>&b, int i, int j)
{
	if (i == 0 || j == 0)return;
	else
	{
		if (b[i][j] == 'd'){ 
			printLCS(X, b, i - 1, j - 1);
			cout << X[i-1];
		}
		else if (b[i][j] == 'u')printLCS(X,b,i-1,j);
		else printLCS(X,b,i,j-1);
	}
}
int LCS_LESS_SPACE(const vector<char>&X, const vector<char>&Y, vector<vector<char>>&b)
{
	
	int m = X.size();//X:<x0,x1....x(m-1)>
	int n = Y.size();
	b = vector<vector<char>>(m + 1, vector<char>(n + 1));
	vector<int>upperRow(n+1,0);
	vector<int>curRow(n + 1, 0);
	for (int i = 1; i <= m; ++i)
	{
		int leftVal = 0;
		int curVal;
		for (int j = 1; j <= n; ++j)
		{
			if (X[i - 1] == Y[j - 1])
			{
				curVal = upperRow[j - 1]+1;
				b[i][j] = 'd';
			}
			else
			{
				if (leftVal > upperRow[j])
				{
					curVal = leftVal;
					b[i][j] = 'l';
				}
				else
				{
					curVal = upperRow[j];
					b[i][j] = 'u';
				}
			}
			leftVal = curVal;
			curRow[j] = curVal;
		}
		upperRow = curRow;
	}
	return upperRow.back();
}
void main()
{
	
	//vector<int>arr = { 13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,-1000 };
	//vector<int>arr2 = { 13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7, -1000 };
	//vector<string>arrStr = {"100","223","034","311","109"};
	//radixSort(arrStr,3);              
	//mergeSort(arr);
	//cout << orderSelection(arr2, 0, arr2.size() - 1, 16) << endl;
	//priority_queue<int,vector<int>,greater<int>>pq(arr.begin(),arr.end());
	//while (!pq.empty())
	//{
	//	cout << pq.top() << " ";
	//  pq.pop();
	//}
	//vector<int>arr = {30,35,15,5,10,20,25};
	//vector<vector<int>>s;
	//cout << "min operation times:" << memoizedMatrixChain(arr, s) << endl;
	//printOptimalParens(s,1,6);
	vector<char>X = {'A','C','C','G','G','T','C','G','A','G','T','G','C','G','C','G','G','A','A','G','C','C','G','G','C','C','G','A','A'};
	string y = "GTCGTTCGGAATGCCGTTGCTCTGTAAA";
	vector<char>Y;
	for (auto yy : y)
		Y.push_back(yy);
	
	vector<vector<char>>b;
	cout << LCS_LESS_SPACE(X, Y, b) << endl;
	printLCS(X,b,X.size(),Y.size());
	int axxx = 0; 
}