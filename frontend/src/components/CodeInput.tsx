import React, { useState } from "react";
import { useEffect } from "react";
// import hljs from "highlight.js";
// import "highlight.js/styles/github-dark.css";

import "../App.css";

const CodeInput = ({ onChange }) => {
  useEffect(() => {
    // hljs.highlightAll();
  }, []);
  return (
    <>
      <textarea onChange={onChange} placeholder="//paste code here"></textarea>
    </>
  );
};

export default CodeInput;
