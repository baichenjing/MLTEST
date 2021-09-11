class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        total_len = len(nums1) + len(nums2)


        total_num=[0]*total_len
        first=0
        second=0
        k=0
        while first < len(nums1) and second < len(nums2):
            if nums1[first] < nums2[second]:
                total_num[k]=nums1[first]
                first = first + 1
            else:
                total_num[k] = nums2[second]
                second = second + 1
            k=k+1
        while first<len(nums1):
            total_num[k]=nums1[first]
            first=first+1
            k=k+1
        while second < len(nums2):
            total_num[k] = nums2[second]
            second = second + 1
            k = k + 1

        if total_len % 2 == 0:
            media_num = (total_len // 2)-1
            return float((total_num[media_num]+total_num[media_num+1])/2)
        else:
            media_num= (total_len // 2)
            return float(total_num[media_num])

if __name__ == '__main__':
    nums1 = [1, 3]
    nums2 = [2,5]
    print(Solution().findMedianSortedArrays(nums1, nums2))