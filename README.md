# od_work

## target data
open image v4からタクシーのみ取得。
train/test合わせて、1502枚を取得。
バウンディングボックスをVOTTを用いてアノテーション。
756枚の画像に1567個のBB。746枚(1502-756)は識別対象が0。
画像サイズは、w,h = 2400, 1600でまずは実施。  
