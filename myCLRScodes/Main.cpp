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
#include <algorithm>
#include <tuple>
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
	for (int i = 1; i < arr.size();++i)
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
	for (int i = 1; i < cntArr.size(); ++i)
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
void main()
{
	vector<int>arr = { 13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,-1000 };
	vector<int>arr2 = { 13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7, -1000 };
	//vector<string>arrStr = {"100","223","034","311","109"};
	//radixSort(arrStr,3);
	//vector<int>arr = {};
	mergeSort(arr);
	cout << orderSelection(arr2, 0, arr2.size() - 1, 16) << endl;
	int axxx = 0;
}