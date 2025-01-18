import React, { useState } from "react";
import { useEffect } from "react";

import "../App.css";

const CodeInput = ({ onChange }: any) => {
  return (
    <>
      <textarea onChange={onChange} placeholder="//paste code here"></textarea>
    </>
  );
};

export default CodeInput;
