import React from 'react';
import Header from './Header';
import ImageGallery from './Images'; 
function LandingPage() {
  return (
    <div>
      <Header />
      <ImageGallery />
      {/* その他のランディングページコンテンツ */}
    </div>
  );
}

export default LandingPage;