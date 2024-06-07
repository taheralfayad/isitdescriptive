<?php

namespace Taheralfayad\IsItDescriptive;

class DescriptivenessModel {
    public function run($text, $actual_label) {
        $escaped_text = escapeshellarg($text);
        $escaped_label = escapeshellarg($actual_label);
        return shell_exec("python3 model.py $escaped_text $escaped_label");
    }
}
