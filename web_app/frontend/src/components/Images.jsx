import React, { useState, useEffect } from 'react';

function ImageGallery() {
  const [imageId, setImageId] = useState(1); // 現在の画像ID
  const [imageData, setImageData] = useState(null); // 画像データ

  // 画像データをAPIから取得する関数
  const fetchImageData = async (id) => {
    try {
      const response = await fetch(`http://localhost:8000/images/${id}`);
      const data = await response.json();
      setImageData(data);
    } catch (error) {
      console.error("画像の取得に失敗しました:", error);
    }
  };

  // 画像IDが変更されたときに実行
  useEffect(() => {
    fetchImageData(imageId);
  }, [imageId]);

  // 次の画像へ進む
  const handleNext = () => {
    setImageId((prevId) => prevId + 1);
  };

  // 前の画像へ戻る
  const handlePrev = () => {
    setImageId((prevId) => prevId - 1);
  };

  return (
    <div>
      <h1>Cell viewer</h1>
      {imageData && (
        <div>
          <img src={`data:image/jpeg;base64,${imageData.originalImage}`} alt="Original" />
          <img src={`data:image/jpeg;base64,${imageData.processedImage}`} alt="Processed" />
        </div>
      )}
      <button onClick={handlePrev}>前へ</button>
      <button onClick={handleNext}>次へ</button>
    </div>
  );
}

export default ImageGallery;
